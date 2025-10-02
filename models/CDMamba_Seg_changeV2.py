from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from models.mamba_customer import ConvMamba


# ---------- Positional Encoding ----------
class ConvPosEnc(nn.Module):
    def __init__(self, dim, k=3):
        super().__init__()
        padding = k // 2
        self.proj = nn.Conv2d(dim, dim, kernel_size=k, padding=padding, groups=dim, bias=True)

    def forward(self, x):  # x: [B,C,H,W]
        return x + self.proj(x)


# ---------- SRCM (with better mixing; item #6) ----------
class ModifiedSRCMLayer(nn.Module):
    """
    - Pre-mix (1x1 conv) on BCHW, then GroupNorm on BCHW (robust stats), then tokenize.
    - Keep LayerNorm on tokens (stability for Mamba).
    - Grouped ConvMamba, then post-mix (1x1 conv) on BCHW.
    """
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2, groups=4, gn_groups=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.groups = groups

        # BCHW pre/post mixers
        self.pre_mix_2d = nn.Conv2d(input_dim, input_dim, kernel_size=1, bias=False)
        self.pre_gn = nn.GroupNorm(num_groups=min(gn_groups, input_dim), num_channels=input_dim)
        self.pos_enc = ConvPosEnc(input_dim)

        # token norm (keep LN for sequence)
        self.token_ln = nn.LayerNorm(input_dim)

        # grouped mamba
        c_per = input_dim // groups
        assert c_per * groups == input_dim, "input_dim must be divisible by groups for grouped Mamba."
        self.mambas = nn.ModuleList([
            ConvMamba(d_model=c_per, d_state=d_state, d_conv=d_conv, expand=expand, bimamba_type="v2")
            for _ in range(groups)
        ])

        self.gate_proj = nn.Linear(input_dim, input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

        self.post_mix_2d = nn.Conv2d(output_dim, output_dim, kernel_size=1, bias=False)

    def forward(self, x):  # x: [B,C,H,W]
        B, C, H, W = x.shape

        # conv positional enc + pre-mix + GN (item #6)
        x = self.pos_enc(x)
        x = self.pre_mix_2d(x)
        x = self.pre_gn(x)

        # BCHW -> tokens
        x_tokens = x.flatten(2).transpose(1, 2)  # [B,HW,C]
        x_norm = self.token_ln(x_tokens)

        # grouped mamba
        chunks = x_norm.chunk(self.groups, dim=-1)
        out_chunks = [m(chunk) for m, chunk in zip(self.mambas, chunks)]
        x_mamba = torch.cat(out_chunks, dim=-1)

        # gated residual in token space
        gate = torch.sigmoid(self.gate_proj(x_norm))
        x_out = gate * x_mamba + (1 - gate) * x_tokens

        # project back to channels and reshape
        x_out = self.proj(x_out).transpose(1, 2).reshape(B, self.output_dim, H, W)

        # post-mix in BCHW
        x_out = self.post_mix_2d(x_out)
        return x_out


def get_srcm_layer(spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1, conv_mode: str = "deepwise"):
    layer = ModifiedSRCMLayer(input_dim=in_channels, output_dim=out_channels)
    if stride != 1:
        if spatial_dims == 2:
            return nn.Sequential(layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
    return layer


# ---------- Residual SRCM Block (SE/Dropout order fixed; item #10) ----------
class SRCMBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: int = 3,
        conv_mode: str = "deepwise",
        act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_srcm_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels, conv_mode=conv_mode)
        self.conv2 = get_srcm_layer(spatial_dims, in_channels=in_channels, out_channels=in_channels, conv_mode=conv_mode)
        self.res_scale = nn.Parameter(torch.tensor(1.0))
        self.drop = nn.Dropout2d(p=0.1)
        # SE computed on pre-drop activations, applied after dropout (item #10)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels // 8), 1), nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // 8), in_channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        s = self.se(x)         # compute SE before dropout (stable stats)
        x = self.drop(x)       # then dropout
        x = x * s              # then apply SE weights
        return identity + self.res_scale * x


# ---------- Main Model ----------
class CDMamba_seg_cd(nn.Module):
    """
    Mamba-based Segmentation + Change Detection with:
    - Explicit change decoder fed by multi-scale |Δ|-pyramid (item #1)
    - Multi-scale (FPN-like) change head (item #2)
    - Learned upsampling in last 2 stages (item #7)
    - Robust skip pairing by shape (item #8)
    - Per-level GN before differencing for illumination robustness (item #12)
    - Higher base capacity default (item #13)
    - Expose multi-scale features for consistency losses (item #11 integration support)
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        init_filters: int = 32,  # item #13: stronger default
        in_channels: int = 1,
        num_classes: int = 7,
        use_change_head: bool = True,
        conv_mode: str = "deepwise",
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        up_conv_mode: str = "deepwise",
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
        learned_last_n_ups: int = 2,  # item #7
        return_pyramids: bool = False # item #11 (to expose multi-scale feats for losses)
    ):
        super().__init__()
        self.num_classes = num_classes
        self.use_change_head = use_change_head
        self.return_pyramids = return_pyramids

        if spatial_dims not in (2, 3):
            raise ValueError("`spatial_dims` can only be 2 or 3.")
        self.up_conv_mode = up_conv_mode
        self.conv_mode = conv_mode
        self.spatial_dims = spatial_dims
        self.init_filters = init_filters
        self.in_channels = in_channels
        self.blocks_down = blocks_down
        self.blocks_up = blocks_up
        self.dropout_prob = dropout_prob
        self.act = act
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.learned_last_n_ups = learned_last_n_ups

        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)

        # encoder / decoders
        self.srcm_encoder_layers = self._make_srcm_encoder_layers()
        self.srcm_decoder_layers, self.up_samples = self._make_srcm_decoder_layers()
        self.srcm_decoder_layers_seg_t1, self.up_samples_seg_t1 = self._make_srcm_decoder_layers()
        self.srcm_decoder_layers_seg_t2, self.up_samples_seg_t2 = self._make_srcm_decoder_layers()
        # change decoder (item #1)
        self.srcm_decoder_layers_chg, self.up_samples_chg = self._make_srcm_decoder_layers()

        # per-level GN for robust differencing (item #12)
        self.diff_norms = nn.ModuleList([
            nn.GroupNorm(num_groups=min(8, self.init_filters * 2 ** i), num_channels=self.init_filters * 2 ** i)
            for i in range(len(self.blocks_down))
        ])

        # ---- Bottleneck context (parallel ASPP-lite; item #2 support) ----
        bottleneck_channels = init_filters * (2 ** (len(blocks_down) - 1))
        if spatial_dims == 2:
            Conv = nn.Conv2d
            Pool = nn.AdaptiveAvgPool2d
        else:
            Conv = nn.Conv3d
            Pool = nn.AdaptiveAvgPool3d

        # parallel branches instead of sequential pool-in-path (fix of original)
        self.aspp_b1 = nn.Sequential(
            Conv(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=bottleneck_channels),
            self.act_mod,
        )
        self.aspp_b2 = nn.Sequential(
            Conv(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=bottleneck_channels),
            self.act_mod,
        )
        self.aspp_b3_pool = Pool(1)
        self.aspp_b3_conv = Conv(bottleneck_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.aspp_fuse = Conv(bottleneck_channels * 3, bottleneck_channels, kernel_size=1, bias=False)

        # --- SEGMENTATION HEADS ---
        self.seg_head_t1 = nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, self.num_classes, kernel_size=1, bias=True),
        )
        self.seg_head_t2 = nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, self.num_classes, kernel_size=1, bias=True),
        )

        # Multi-scale change head (tiny FPN over last 3 decoder stages; item #2)
        if self.use_change_head:
            # unify to init_filters
            self.chg_lateral_3 = nn.Conv2d(self.init_filters, self.init_filters, 1, bias=False)  # finest
            self.chg_lateral_2 = nn.Conv2d(self.init_filters * 2, self.init_filters, 1, bias=False)
            self.chg_lateral_1 = nn.Conv2d(self.init_filters * 4, self.init_filters, 1, bias=False)
            self.chg_out = nn.Sequential(
                nn.Conv2d(self.init_filters, self.init_filters, 3, padding=1, bias=False),
                get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
                self.act_mod,
                nn.Conv2d(self.init_filters, 2, 1, bias=True),
            )

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    # ---------- helpers ----------
    def _context(self, x):
        b1 = self.aspp_b1(x)
        b2 = self.aspp_b2(x)
        b3 = self.aspp_b3_conv(self.aspp_b3_pool(x))
        b3 = F.interpolate(b3, size=x.shape[2:], mode='bilinear', align_corners=False) if self.spatial_dims == 2 else F.interpolate(b3, size=x.shape[2:], mode='trilinear', align_corners=False)
        return self.aspp_fuse(torch.cat([b1, b2, b3], dim=1))

    def _make_srcm_encoder_layers(self):
        layers = nn.ModuleList()
        filters = self.init_filters
        for i, item in enumerate(self.blocks_down):
            ch = filters * 2 ** i
            downsample_mamba = get_srcm_layer(self.spatial_dims, ch // 2, ch, stride=2, conv_mode=self.conv_mode) if i > 0 else nn.Identity()
            block_seq = nn.Sequential(
                downsample_mamba,
                *[SRCMBlock(self.spatial_dims, ch, norm=self.norm, act=self.act, conv_mode=self.conv_mode) for _ in range(item)]
            )
            layers.append(block_seq)
        return layers

    def _make_srcm_decoder_layers(self):
        layers, ups = nn.ModuleList(), nn.ModuleList()
        filters = self.init_filters
        n_up = len(self.blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)        # 128, 64, 32 if init=32 & n_up=3
            cat_channels = (sample_in_channels // 2) * 2          # concat of up and skip

            layers.append(nn.Sequential(
                get_conv_layer(self.spatial_dims, cat_channels, sample_in_channels // 2, kernel_size=3, stride=1),
                SRCMBlock(self.spatial_dims, sample_in_channels // 2, norm=self.norm, act=self.act, conv_mode=self.up_conv_mode)
            ))

            # upsample: learned for last N (item #7)
            if i >= n_up - self.learned_last_n_ups:
                if self.spatial_dims == 2:
                    ups.append(nn.ConvTranspose2d(sample_in_channels, sample_in_channels // 2, kernel_size=2, stride=2))
                else:
                    ups.append(nn.ConvTranspose3d(sample_in_channels, sample_in_channels // 2, kernel_size=2, stride=2))
            else:
                ups.append(nn.Sequential(
                    get_conv_layer(self.spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                    get_upsample_layer(self.spatial_dims, sample_in_channels // 2, upsample_mode=self.upsample_mode),
                ))
        return layers, ups

    @staticmethod
    def _pair_by_shape(x_up: torch.Tensor, skip_list: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Robustly pick the first skip whose spatial dims match x_up; pop it (item #8).
        """
        for idx, s in enumerate(skip_list):
            if s.shape[2:] == x_up.shape[2:]:
                skip = s
                del skip_list[idx]
                return skip, skip_list
        # fallback: take nearest by size (minimal absolute difference)
        sizes = [torch.tensor(s.shape[2:]) for s in skip_list]
        diffs = [torch.abs(sz - torch.tensor(x_up.shape[2:])).sum().item() for sz in sizes]
        j = int(torch.tensor(diffs).argmin())
        skip = skip_list[j]
        del skip_list[j]
        return skip, skip_list

    def _decode_with_layers(self, x: torch.Tensor, skips: list[torch.Tensor],
                            up_samples: nn.ModuleList, decoder_layers: nn.ModuleList,
                            collect_scales: bool = False):
        """
        Generic decoder; `skips` must be a list of encoder features (deep->shallow) WITHOUT the bottleneck.
        Returns final feature and (optionally) a list of intermediate outputs (coarse->fine).
        """
        pyr = []
        # we will not mutate the original list outside
        skip_pool = list(skips)
        for up, block in zip(up_samples, decoder_layers):
            x_up = up(x)
            # find matching skip by shape
            skip, skip_pool = self._pair_by_shape(x_up, skip_pool)
            x = torch.cat([x_up, skip], dim=1)
            x = block(x)
            if collect_scales:
                pyr.append(x)
        return (x, pyr) if collect_scales else x

    def encode(self, x: torch.Tensor):
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        enc_feats = []
        for enc in self.srcm_encoder_layers:
            x = enc(x)
            enc_feats.append(x)
        return x, enc_feats  # x is bottleneck-in, enc_feats includes all levels

    def _build_skip_list(self, enc_feats: list[torch.Tensor]):
        """
        Build deep->shallow skip list excluding the deepest (bottleneck) feature.
        """
        return list(reversed(enc_feats[:-1]))

    def _multi_scale_change_head(self, pyr_feats):
        """
        FPN over last 3 scales (coarse->fine order in pyr_feats).
        We expect pyr_feats[-1] ~ finest (C=self.init_filters).
        """
        # ensure we have at least 3; if fewer, repeat coarsest
        if len(pyr_feats) < 3:
            while len(pyr_feats) < 3:
                pyr_feats = [pyr_feats[0]] + pyr_feats
        f1, f2, f3 = pyr_feats[-3], pyr_feats[-2], pyr_feats[-1]  # coarse, mid, fine

        l1 = self.chg_lateral_1(f1)  # -> init_filters
        l2 = self.chg_lateral_2(f2)  # -> init_filters
        l3 = self.chg_lateral_3(f3)  # -> init_filters

        # top-down upsample & sum
        u2 = F.interpolate(l1, size=l2.shape[2:], mode='bilinear', align_corners=False) + l2
        u3 = F.interpolate(u2, size=l3.shape[2:], mode='bilinear', align_corners=False) + l3

        return self.chg_out(u3)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None):
        """
        Returns:
            Single image:
                seg_logits_t1
                (optionally) {'pyr_t1': [...]} if return_pyramids=True
            Bi-temporal:
                seg_logits_t1, seg_logits_t2, change_logits
                (optionally) dict with pyramids for consistency losses (item #11)
        """
        # ----- Single-image segmentation -----
        if x2 is None:
            latent, feats = self.encode(x1)
            latent = self._context(latent)
            skips = self._build_skip_list(feats)
            dec, pyr = self._decode_with_layers(latent, skips, self.up_samples, self.srcm_decoder_layers, collect_scales=self.return_pyramids)
            seg_logits_t1 = self.seg_head_t1(dec)
            if self.return_pyramids:
                return seg_logits_t1, {"pyr_t1": pyr}
            return seg_logits_t1

        # ----- Bi-temporal mode -----
        x1_latent, feats1 = self.encode(x1)
        x2_latent, feats2 = self.encode(x2)

        x1_latent = self._context(x1_latent)
        x2_latent = self._context(x2_latent)

        # per-level normalized difference pyramid (item #12)
        # feats lists are shallow->deep; we want deep->shallow skips excluding deepest
        # but we need Δ at all encoder levels including deepest to seed change latent
        diffs = []
        for i, (f1, f2) in enumerate(zip(feats1, feats2)):
            g1 = self.diff_norms[i](f1)
            g2 = self.diff_norms[i](f2)
            diffs.append(torch.abs(g1 - g2))

        # latent |Δ| (use deepest, i = last)
        x_delta_latent = torch.abs(self.diff_norms[-1](x1_latent) - self.diff_norms[-1](x2_latent))
        x_delta_latent = self._context(x_delta_latent)

        # build skip lists (deep->shallow) for each branch
        skips1 = self._build_skip_list(feats1)
        skips2 = self._build_skip_list(feats2)
        skips_delta = self._build_skip_list(diffs)  # already aligned by level

        # decode each stream; collect pyramids for change FPN & losses
        seg1, pyr1 = self._decode_with_layers(x1_latent, skips1, self.up_samples_seg_t1, self.srcm_decoder_layers_seg_t1, collect_scales=True)
        seg2, pyr2 = self._decode_with_layers(x2_latent, skips2, self.up_samples_seg_t2, self.srcm_decoder_layers_seg_t2, collect_scales=True)
        chg, pyr_delta = self._decode_with_layers(x_delta_latent, skips_delta, self.up_samples_chg, self.srcm_decoder_layers_chg, collect_scales=True)

        seg_logits_t1 = self.seg_head_t1(seg1)
        seg_logits_t2 = self.seg_head_t2(seg2)

        if self.use_change_head:
            # multi-scale change head over delta decoder pyramid (item #2)
            change_logits = self._multi_scale_change_head(pyr_delta)
            if self.return_pyramids:
                return (seg_logits_t1, seg_logits_t2, change_logits,
                        {"pyr_t1": pyr1, "pyr_t2": pyr2, "pyr_delta": pyr_delta})
            return seg_logits_t1, seg_logits_t2, change_logits
        else:
            if self.return_pyramids:
                return (seg_logits_t1, seg_logits_t2,
                        {"pyr_t1": pyr1, "pyr_t2": pyr2})
            return seg_logits_t1, seg_logits_t2


# ----------------- quick sanity test -----------------
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = CDMamba_seg_cd(
        spatial_dims=2,
        in_channels=3,
        num_classes=6,
        use_change_head=True,
        init_filters=32,                 # item #13 stronger default
        up_conv_mode="deepwise",
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        learned_last_n_ups=2,            # item #7
        return_pyramids=True             # item #11: expose multi-scale feats for losses
    ).to(device)

    # Single-image
    x = torch.randn(1, 3, 256, 256).to(device)
    seg, aux = model(x)
    print("Single image seg:", seg.shape, "pyr len:", len(aux["pyr_t1"]))

    # Bi-temporal
    x1 = torch.randn(1, 3, 256, 256).to(device)
    x2 = torch.randn(1, 3, 256, 256).to(device)
    seg_t1, seg_t2, change, aux = model(x1, x2)
    print("Bi-temporal:")
    print("  seg_t1:", seg_t1.shape, " seg_t2:", seg_t2.shape, " change:", change.shape)
    print("  pyrs:", len(aux["pyr_t1"]), len(aux["pyr_t2"]), len(aux["pyr_delta"]))
