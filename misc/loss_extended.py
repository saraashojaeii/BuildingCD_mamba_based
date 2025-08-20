
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- small utils ----------

def _resize_like(inp: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear"):
    \"\"\"Resize logits/probs inp to have the same HxW as ref (no-op if already same).\"\"\"
    if inp.shape[-2:] != ref.shape[-2:]:
        return F.interpolate(inp, size=ref.shape[-2:], mode=mode, align_corners=(mode=='bilinear'))
    return inp

# ---------- base pieces ----------

class BinaryDiceLoss(nn.Module):
    \"\"\"Soft Dice on a binary mask, taking logits as input by default.\"\"\"
    def __init__(self, from_logits: bool = True, smooth: float = 1e-6):
        super().__init__()
        self.from_logits = from_logits
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # pred: [B,1,H,W] (logits or probs), target: [B,1,H,W] or [B,H,W] in {0,1}
        if target.dim() == 3:
            target = target.unsqueeze(1)
        pred = _resize_like(pred, target)
        if self.from_logits:
            prob = torch.sigmoid(pred)
        else:
            prob = pred.clamp_min(0).clamp_max(1)

        target = target.float()
        intersection = (prob * target).sum(dim=(1,2,3))
        union = prob.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

class ChangeHeadBCEDiceLoss(nn.Module):
    \"\"\"BCEWithLogits + lambda_dice * BinaryDiceLoss for the change head.\"\"\"
    def __init__(self, lambda_dice: float = 1.0, pos_weight: float = None):
        super().__init__()
        self.lambda_dice = float(lambda_dice)
        self.dice = BinaryDiceLoss(from_logits=True)
        self.pos_weight = None if pos_weight is None else torch.tensor([pos_weight])

    def forward(self, change_logits: torch.Tensor, change_gt: torch.Tensor):
        # change_logits: [B,1,H,W], change_gt: [B,1,H,W] or [B,H,W] in {0,1}
        if change_gt.dim() == 3:
            change_gt = change_gt.unsqueeze(1)
        change_logits = _resize_like(change_logits, change_gt)
        bce = F.binary_cross_entropy_with_logits(change_logits, change_gt.float(), pos_weight=self.pos_weight)
        dice = self.dice(change_logits, change_gt)
        return bce + self.lambda_dice * dice

class UnchangedSymmetricKLLoss(nn.Module):
    \"\"\"Symmetric KL between t1/t2 class distributions on UNCHANGED pixels (c==0).
    Inputs are logits for numerical stability; temperature T softens distributions.
    \"\"\"
    def __init__(self, T: float = 4.0, detach_one_side: bool = True, eps: float = 1e-8):
        super().__init__()
        self.T = float(T)
        self.detach_one_side = bool(detach_one_side)
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, change_mask: torch.Tensor):
        # z1,z2: [B,C,H,W] logits; change_mask: [B,1,H,W] or [B,H,W] where 0=unchanged,1=changed
        if change_mask.dim() == 3:
            change_mask = change_mask.unsqueeze(1)
        z1 = _resize_like(z1, change_mask)
        z2 = _resize_like(z2, change_mask)

        unch = (change_mask == 0).float()  # [B,1,H,W]
        if unch.sum() < 1:
            return torch.zeros([], device=z1.device, dtype=z1.dtype)

        z1T = z1 / self.T
        z2T = z2 / self.T
        logp1 = F.log_softmax(z1T, dim=1)
        logp2 = F.log_softmax(z2T, dim=1)
        p1 = F.softmax(z1T, dim=1)
        p2 = F.softmax(z2T, dim=1)
        if self.detach_one_side:
            p1 = p1.detach()
            p2 = p2.detach()

        kl12 = (p1 * (logp1 - logp2)).sum(dim=1, keepdim=True)  # [B,1,H,W]
        kl21 = (p2 * (logp2 - logp1)).sum(dim=1, keepdim=True)
        sym = (kl12 + kl21) * unch
        return sym.sum() / (unch.sum() + self.eps)

class ChangedDiversityCosineMarginLoss(nn.Module):
    \"\"\"Encourage dissimilar semantics on CHANGED pixels via cosine similarity margin.\"\"\"
    def __init__(self, margin: float = 0.3, eps: float = 1e-8):
        super().__init__()
        self.margin = float(margin)
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, change_mask: torch.Tensor):
        # logits -> probs
        if change_mask.dim() == 3:
            change_mask = change_mask.unsqueeze(1)
        z1 = _resize_like(z1, change_mask)
        z2 = _resize_like(z2, change_mask)

        p1 = F.softmax(z1, dim=1)
        p2 = F.softmax(z2, dim=1)
        ch = (change_mask == 1).float()  # [B,1,H,W]
        if ch.sum() < 1:
            return torch.zeros([], device=z1.device, dtype=z1.dtype)

        # cosine similarity along class dim
        num = (p1 * p2).sum(dim=1, keepdim=True)
        den = p1.norm(p=2, dim=1, keepdim=True) * p2.norm(p=2, dim=1, keepdim=True) + self.eps
        s = (num / den)  # [B,1,H,W]
        loss = F.relu(s - self.margin) * ch
        return loss.sum() / (ch.sum() + self.eps)

class CouplingChangeSemanticLoss(nn.Module):
    \"\"\"Penalize disagreement between change prob q and semantic-derived change score r=1-dot(p1,p2).\"\"\"
    def __init__(self, distance: str = "l1", detach_semantics: bool = False, eps: float = 1e-8):
        super().__init__()
        assert distance in {"l1","l2","bce"}
        self.distance = distance
        self.detach_semantics = detach_semantics
        self.eps = eps

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, change_logits: torch.Tensor):
        # z1,z2: [B,C,H,W], change_logits: [B,1,H,W]
        change_logits = change_logits
        z1 = _resize_like(z1, change_logits)
        z2 = _resize_like(z2, change_logits)

        p1 = F.softmax(z1, dim=1)
        p2 = F.softmax(z2, dim=1)
        if self.detach_semantics:
            p1 = p1.detach(); p2 = p2.detach()

        q = torch.sigmoid(change_logits)           # [B,1,H,W]
        e = (p1 * p2).sum(dim=1, keepdim=True)     # probability of equality
        r = 1.0 - e                                # semantic change score in [0,1]

        if self.distance == "l1":
            return (q - r).abs().mean()
        elif self.distance == "l2":
            return ((q - r) ** 2).mean()
        else:  # 'bce' view: treat r as soft target for q
            r = r.clamp(min=0.0 + self.eps, max=1.0 - self.eps)
            return F.binary_cross_entropy(q, r)

# ---------- example composer (optional) ----------

class TripletChangeSegLoss(nn.Module):
    \"\"\"Convenience wrapper that sums:
        - seg losses t1 & t2 (you provide a callable seg_loss_fn)
        - change loss (BCE+Dice)
        - unchanged symmetric KL
        - changed diversity (cos margin)
        - coupling loss
    You can also use each class independently in your train script.
    \"\"\"
    def __init__(self,
                 seg_loss_fn: nn.Module,
                 lambda_seg: float = 1.0,
                 lambda_cd: float = 1.0,
                 lambda_unch: float = 0.2,
                 lambda_ch: float = 0.2,
                 lambda_cpl: float = 0.5,
                 T: float = 4.0,
                 margin: float = 0.3):
        super().__init__()
        self.seg_loss_fn = seg_loss_fn
        self.cd_loss = ChangeHeadBCEDiceLoss(lambda_dice=1.0)
        self.unch_kl = UnchangedSymmetricKLLoss(T=T)
        self.ch_div = ChangedDiversityCosineMarginLoss(margin=margin)
        self.couple = CouplingChangeSemanticLoss(distance="l1")

        self.lam_seg = lambda_seg
        self.lam_cd = lambda_cd
        self.lam_unch = lambda_unch
        self.lam_ch = lambda_ch
        self.lam_cpl = lambda_cpl

    def forward(self, preds, targets):
        # preds: (z1, z2, u); targets: dict with keys seg_t1, seg_t2, change
        z1, z2, u = preds
        y1 = targets["seg_t1"]
        y2 = targets["seg_t2"]
        c  = targets["change"]

        # segmentation (uses your own CE/Dice combo)
        L_seg = self.seg_loss_fn(z1, y1) + self.seg_loss_fn(z2, y2)
        # change head
        L_cd = self.cd_loss(u, c)
        # cross-time
        L_unch = self.unch_kl(z1, z2, c)
        L_ch   = self.ch_div(z1, z2, c)
        # coupling
        L_cpl  = self.couple(z1, z2, u)

        total = (self.lam_seg * L_seg +
                 self.lam_cd  * L_cd  +
                 self.lam_unch* L_unch+
                 self.lam_ch  * L_ch  +
                 self.lam_cpl * L_cpl)

        return total, {
            "seg": L_seg.item(),
            "cd": L_cd.item(),
            "unch_kl": L_unch.item(),
            "ch_div": L_ch.item(),
            "couple": L_cpl.item()
        }
