import torch
import torch.nn as nn
from torch import Tensor, einsum
import torch.nn .functional as F
from misc.torchutils import class2one_hot,simplex
from models.darnet_help.loss_help import FocalLoss, dernet_dice_loss

def cross_entropy_loss_fn(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]: # Ensure spatial dimensions match
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

class DiceLoss(nn.Module):
    def __init__(self, num_classes, weight=None, ignore_index=255, smooth=1e-10, idc=None):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.ignore_index = ignore_index 
        self.smooth = smooth
        self.idc = idc if idc is not None else list(range(self.num_classes))

    def forward(self, predicts, target):
        probs = torch.softmax(predicts, dim=1)
        
        target = target.long()
        if target.dim() == 4:
            target = target.squeeze(1)
        
        target_one_hot = class2one_hot(target, self.num_classes)
        
        assert simplex(probs)
        assert simplex(target_one_hot)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target_one_hot[:, self.idc, ...].type(torch.float32)

        intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
        
        card_pc: Tensor = einsum("bcwh->bc", pc)
        card_tc: Tensor = einsum("bcwh->bc", tc)
        
        union: Tensor = card_pc + card_tc

        dice_score_per_class: Tensor = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss_per_class: Tensor = 1.0 - dice_score_per_class

        if self.weight is not None:
            class_weights = torch.tensor(self.weight, device=predicts.device, dtype=torch.float32)[self.idc]
            dice_loss_per_class = dice_loss_per_class * class_weights.view(1, -1) # ensure broadcasting b*c
            loss = (dice_loss_per_class.sum(dim=1) / class_weights.sum()).mean() # weighted mean over classes, then mean over batch
        else:
            loss = dice_loss_per_class.mean() # Mean over classes and batch

        return loss

class CEDiceLoss(nn.Module):
    def __init__(self, num_classes, ce_weight=0.5, dice_weight=0.5, cross_entropy_kwargs=None, dice_loss_kwargs=None):
        super(CEDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        _ce_kwargs = cross_entropy_kwargs if cross_entropy_kwargs is not None else {}
        self.cross_entropy_fn = cross_entropy_loss_fn

        _dice_kwargs = dice_loss_kwargs if dice_loss_kwargs is not None else {}
        self.dice_loss = DiceLoss(num_classes=self.num_classes, **_dice_kwargs)

    def forward(self, input, target):
        ce_loss = self.cross_entropy_fn(input, target)
        dice_val = self.dice_loss(input, target)
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_val
        return loss

class DiceOnlyLoss(nn.Module):
    def __init__(self, num_classes, dice_loss_kwargs=None):
        super(DiceOnlyLoss, self).__init__()
        self.num_classes = num_classes
        _dice_kwargs = dice_loss_kwargs if dice_loss_kwargs is not None else {}
        self.dice_loss = DiceLoss(num_classes=self.num_classes, **_dice_kwargs)

    def forward(self, input, target):
        return self.dice_loss(input, target)

class CE2Dice1Loss(nn.Module):
    def __init__(self, num_classes, ce_weight=1.0, dice_weight=0.5, cross_entropy_kwargs=None, dice_loss_kwargs=None):
        super(CE2Dice1Loss, self).__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        _ce_kwargs = cross_entropy_kwargs if cross_entropy_kwargs is not None else {}
        self.cross_entropy_fn = cross_entropy_loss_fn

        _dice_kwargs = dice_loss_kwargs if dice_loss_kwargs is not None else {}
        self.dice_loss = DiceLoss(num_classes=self.num_classes, **_dice_kwargs)

    def forward(self, input, target):
        ce_loss = self.cross_entropy_fn(input, target)
        dice_val = self.dice_loss(input, target)
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_val
        return loss

class CE1Dice2Loss(nn.Module):
    def __init__(self, num_classes, ce_weight=0.5, dice_weight=1.0, cross_entropy_kwargs=None, dice_loss_kwargs=None):
        super(CE1Dice2Loss, self).__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        _ce_kwargs = cross_entropy_kwargs if cross_entropy_kwargs is not None else {}
        self.cross_entropy_fn = cross_entropy_loss_fn

        _dice_kwargs = dice_loss_kwargs if dice_loss_kwargs is not None else {}
        self.dice_loss = DiceLoss(num_classes=self.num_classes, **_dice_kwargs)
        
    def forward(self, input, target):
        ce_loss = self.cross_entropy_fn(input, target)
        dice_val = self.dice_loss(input, target)
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_val
        return loss

# Note: ce_scl was identical to ce_dice. If it needs specific SCL logic, it requires a separate implementation.
# For now, if 'ce_scl' is chosen, it would need to be mapped to CEDiceLoss or a new SCL specific class.


def weighted_BCE_logits(logit_pixel, truth_pixel, weight_pos=0.25, weight_neg=0.75):
    logit = logit_pixel.view(-1)
    truth = truth_pixel.view(-1)
    assert (logit.shape == truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit.float(), truth.float(), reduction='none')

    pos = (truth > 0.5).float()
    neg = (truth < 0.5).float()
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos * pos * loss / pos_num + weight_neg * neg * loss / neg_num).sum()

    return loss

class ChangeSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """

    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)

    def forward(self, x1, x2, label_change):
        b, c, h, w = x1.size()
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = torch.reshape(x1, [b * h * w, c])
        x2 = torch.reshape(x2, [b * h * w, c])

        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target, [b * h * w])

        loss = self.loss_f(x1, x2, target)
        return loss

def hybrid_loss(predictions, target, weight=[0,2,0.2,0.2,0.2,0.2]):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    # focal = FocalLoss(gamma=0, alpha=None)
    # ssim = SSIM()

    for i,prediction in enumerate(predictions):

        bce = cross_entropy(prediction, target)
        dice = dice_loss(prediction, target)
        # ssimloss = ssim(prediction, target)
        loss += weight[i]*(bce + dice) #- ssimloss

    return loss

class BCL(nn.Module):
    """
    batch-balanced contrastive loss
    no-change，1
    change，-1
    """
    def __init__(self, margin=2.0):
        super(BCL, self).__init__()
        self.margin = margin

    def forward(self, distance, label):
        label[label == 1] = -1
        label[label == 0] = 1

        mask = (label != 255).float()
        distance = distance * mask

        pos_num = torch.sum((label==1).float())+0.0001
        neg_num = torch.sum((label==-1).float())+0.0001

        loss_1 = torch.sum((1+label) / 2 * torch.pow(distance, 2)) /pos_num
        loss_2 = torch.sum((1-label) / 2 *
            torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        ) / neg_num
        loss = loss_1 + loss_2
        return loss