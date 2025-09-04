from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.segresnet_block import get_conv_layer, get_upsample_layer
from monai.networks.layers.factories import Dropout
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import UpsampleMode
from models.mamba_customer import ConvMamba


# removed unused get_dwconv_layer

class ModifiedSRCMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2, groups=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.groups = groups
        self.norm = nn.LayerNorm(input_dim)

        # Grouped ConvMamba (split channels across groups)
        self.mambas = nn.ModuleList([
            ConvMamba(d_model=input_dim // groups, d_state=d_state, d_conv=d_conv, expand=expand, bimamba_type="v2")
            for _ in range(groups)
        ])

        self.gate_proj = nn.Linear(input_dim, input_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, 4096, input_dim))  # Max 32x32 tokens (safe default)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)  # [B, N, C]
        pos_embed = F.interpolate(
            self.pos_embed.transpose(1, 2).reshape(1, self.input_dim, int(self.pos_embed.shape[1] ** 0.5), -1),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).reshape(1, self.input_dim, -1).transpose(1, 2)  # Shape: [1, H*W, C]
        x = x + pos_embed[:, :x.shape[1], :]

        x_norm = self.norm(x)

        # Grouped Mamba
        chunks = x_norm.chunk(self.groups, dim=-1)
        out_chunks = [m(chunk) for m, chunk in zip(self.mambas, chunks)]
        x_mamba = torch.cat(out_chunks, dim=-1)

        # Gated residual
        gate = torch.sigmoid(self.gate_proj(x_norm))
        x_out = gate * x_mamba + (1 - gate) * x

        x_out = self.proj(x_out)
        return x_out.transpose(1, 2).reshape(B, self.output_dim, H, W)

def get_srcm_layer(
        spatial_dims: int, in_channels: int, out_channels: int, stride: int = 1, conv_mode: str = "deepwise"
):
    srcm_layer = ModifiedSRCMLayer(input_dim=in_channels, output_dim=out_channels)  # Removed conv_mode
    if stride != 1:
        if spatial_dims == 2:
            return nn.Sequential(srcm_layer, nn.MaxPool2d(kernel_size=stride, stride=stride))
    return srcm_layer

# removed DirectionalAGLGF (not needed for segmentation)

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
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")
        # print(conv_mode)
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_srcm_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, conv_mode=conv_mode
        )
        self.conv2 = get_srcm_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, conv_mode=conv_mode
        )

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity

        return x

# removed multi-branch fusion classes (L_GF, G_GF, AdaptiveGate) for segmentation-only

class CDMamba_seg(nn.Module):
    """
    Segmentation-only Mamba-based Model.
    - Single input image -> per-pixel segmentation logits (num_classes channels).
    """

    def __init__(
            self,
            spatial_dims: int = 3,
            init_filters: int = 16,
            in_channels: int = 1,
            num_classes: int = 7,
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
    ):
        super().__init__()
        self.num_classes = num_classes

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
        self.act = act  # input options
        self.act_mod = get_act_layer(act)
        if norm_name:
            if norm_name.lower() != "group":
                raise ValueError(f"Deprecating option 'norm_name={norm_name}', please use 'norm' instead.")
            norm = ("group", {"num_groups": num_groups})
        self.norm = norm
        self.upsample_mode = UpsampleMode(upsample_mode)
        self.convInit = get_conv_layer(spatial_dims, in_channels, init_filters)
        self.srcm_encoder_layers = self._make_srcm_encoder_layers()
        self.srcm_decoder_layers, self.up_samples = self._make_srcm_decoder_layers()
        # --- SEGMENTATION HEAD ---
        self.seg_head = nn.Sequential(
            get_norm_layer(name=self.norm, spatial_dims=self.spatial_dims, channels=self.init_filters),
            self.act_mod,
            get_conv_layer(self.spatial_dims, self.init_filters, self.num_classes, kernel_size=1, bias=True),
        )

        if dropout_prob is not None:
            self.dropout = Dropout[Dropout.DROPOUT, spatial_dims](dropout_prob)

    def _make_srcm_encoder_layers(self):
        srcm_encoder_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm, conv_mode = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm, self.conv_mode)
        for i, item in enumerate(blocks_down):
            layer_in_channels = filters * 2 ** i
            downsample_mamba = (
                get_srcm_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2, conv_mode=conv_mode)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                downsample_mamba,
                *[SRCMBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act, conv_mode=conv_mode) for _ in range(item)]
            )
            srcm_encoder_layers.append(down_layer)
        return srcm_encoder_layers

    def _make_srcm_decoder_layers(self):
        srcm_decoder_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        Block_up = SRCMBlock
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = filters * 2 ** (n_up - i)
            srcm_decoder_layers.append(
                nn.Sequential(
                    *[
                        Block_up(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act, conv_mode=self.up_conv_mode)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return srcm_decoder_layers, up_samples

    # removed _make_final_conv (not used in segmentation-only)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.convInit(x)
        if self.dropout_prob is not None:
            x = self.dropout(x)
        down_x = []

        for down in self.srcm_encoder_layers:
            x = down(x)
            down_x.append(x)

        return x, down_x

    def _decode_with_layers(self, x: torch.Tensor, down_x: list[torch.Tensor],
                             up_samples: nn.ModuleList, decoder_layers: nn.ModuleList) -> torch.Tensor:
        """Generic decoder that operates on provided up-sample and decoder layer lists."""
        for i, (up, upl) in enumerate(zip(up_samples, decoder_layers)):
            x_up = up(x)
            # Ensure spatial dimensions match for skip connection
            target_size = down_x[i + 1].shape[2:]
            if x_up.shape[2:] != target_size:
                x_up = F.interpolate(x_up, size=target_size, mode='bilinear', align_corners=False)
            x = x_up + down_x[i + 1]
            x = upl(x)
        return x

    def decode(self, x: torch.Tensor, down_x: list[torch.Tensor]) -> torch.Tensor:
        return self._decode_with_layers(x, down_x, self.up_samples, self.srcm_decoder_layers)


    def forward(self, x: torch.Tensor):
        """
        Returns:
            seg_logits: [B, num_classes, H, W] -- segmentation logits
        """
        latent, down_x = self.encode(x)
        down_x.reverse()
        dec = self._decode_with_layers(latent, down_x, self.up_samples, self.srcm_decoder_layers)
        seg_logits = self.seg_head(dec)
        return seg_logits

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = CDMamba_seg(
        spatial_dims=2,
        in_channels=3,
        num_classes=6,
        init_filters=16,
        up_conv_mode="deepwise",
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    ).to(device)

    x = torch.randn(1, 3, 256, 256).to(device)
    seg = model(x)
    print(seg.shape)
