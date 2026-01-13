# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from ultralytics.utils.metrics import OKS_SIGMA
# from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
# from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
# from ultralytics.utils.torch_utils import autocast
#
# from .metrics import bbox_iou, probiou
# from .tal import bbox2dist

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA, WiseIouLoss, wasserstein_loss
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.atss import ATSSAssigner, generate_anchors
from .metrics import bbox_iou, probiou
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist

import math
import numpy as np
# 辅助头部分的代码从魔鬼面具v11代码里面复制，已和魔导沟通确认，具体请参考  https://github.com/z1069614715/objectdetection_script
class SlideLoss(nn.Module):
    def __init__(self, loss_fcn):
        super(SlideLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply SL to each element

    def forward(self, pred, true, auto_iou=0.5):
        loss = self.loss_fcn(pred, true)
        if auto_iou < 0.2:
            auto_iou = 0.2
        b1 = true <= auto_iou - 0.1
        a1 = 1.0
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
        a2 = math.exp(1.0 - auto_iou)
        b3 = true >= auto_iou
        a3 = torch.exp(-(true - 1.0))
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss *= modulating_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class EMASlideLoss:
    def __init__(self, loss_fcn, decay=0.999, tau=2000):
        super(EMASlideLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply SL to each element
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        self.is_train = True
        self.updates = 0
        self.iou_mean = 1.0

    def __call__(self, pred, true, auto_iou=0.5):
        if self.is_train and auto_iou != -1:
            self.updates += 1
            d = self.decay(self.updates)
            self.iou_mean = d * self.iou_mean + (1 - d) * float(auto_iou.detach())
        auto_iou = self.iou_mean
        loss = self.loss_fcn(pred, true)
        if auto_iou < 0.2:
            auto_iou = 0.2
        b1 = true <= auto_iou - 0.1
        a1 = 1.0
        b2 = (true > (auto_iou - 0.1)) & (true < auto_iou)
        a2 = math.exp(1.0 - auto_iou)
        b3 = true >= auto_iou
        a3 = torch.exp(-(true - 1.0))
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss *= modulating_weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLossProb(nn.Module):
    """
    Focal Loss for inputs that are already probabilities (after sigmoid).
    pred: tensor of shape (B, ...) with values in [0, 1]
    label: tensor of same shape with values in {0, 1}
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """
        Args:
            pred: probability after sigmoid, shape (B, ...) ∈ [0, 1]
            label: binary label, shape (B, ...) ∈ {0, 1}
            gamma: focusing parameter
            alpha: weighting factor for positive class
        """
        # Numerical stability: avoid log(0)
        eps = 1e-7
        pred = torch.clamp(pred, eps, 1.0 - eps)

        # Compute p_t: probability of the true class
        p_t = label * pred + (1 - label) * (1 - pred)  # same as before

        # Modulating factor
        modulating_factor = (1.0 - p_t) ** gamma

        # BCE loss from probabilities
        bce_loss = -(label * torch.log(pred) + (1 - label) * torch.log(1 - pred))

        # Apply focal modulation
        loss = bce_loss * modulating_factor

        # Apply alpha balancing
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss = loss * alpha_factor

        # Same reduction as original: mean over all dims except batch, then sum over batch
        if loss.ndim > 1:
            loss = loss.mean(dim=tuple(range(1, loss.ndim)))  # mean over H, W, etc.
        # return loss.sum()  # or .mean() if you prefer average over batch
        return loss.mean()


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class VarifocalLoss_YOLO(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        """Initialize the VarifocalLoss class."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_score, gt_score):
        """Computes varfocal loss."""

        weight = self.alpha * (pred_score.sigmoid() - gt_score).abs().pow(self.gamma) * (
                    gt_score <= 0.0).float() + gt_score * (gt_score > 0.0).float()
        with torch.cuda.amp.autocast(enabled=False):
            return F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') * weight


class QualityfocalLoss_YOLO(nn.Module):
    def __init__(self, beta=2.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred_score, gt_score, gt_target_pos_mask):
        # negatives are supervised by 0 quality score
        pred_sigmoid = pred_score.sigmoid()
        scale_factor = pred_sigmoid
        zerolabel = scale_factor.new_zeros(pred_score.shape)
        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy_with_logits(pred_score, zerolabel, reduction='none') * scale_factor.pow(
                self.beta)

        scale_factor = gt_score[gt_target_pos_mask] - pred_sigmoid[gt_target_pos_mask]
        with torch.cuda.amp.autocast(enabled=False):
            loss[gt_target_pos_mask] = F.binary_cross_entropy_with_logits(pred_score[gt_target_pos_mask],
                                                                          gt_score[gt_target_pos_mask],
                                                                          reduction='none') * scale_factor.abs().pow(
                self.beta)
        return loss


class FocalLoss_YOLO(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, gamma=1.5, alpha=0.25):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, label):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if self.alpha > 0:
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing DFL losses during training."""

    def __init__(self, reg_max=16) -> None:
        """Initialize the DFL module."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max=16):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

        # NWD
        self.nwd_loss = False
        self.iou_ratio = 0.5  # total_iou_loss = self.iou_ratio * iou_loss + (1 - self.iou_ratio) * nwd_loss

        # WiseIOU
        self.use_wiseiou = False
        if self.use_wiseiou:
            self.wiou_loss = WiseIouLoss(ltype='WIoU', monotonous=False, inner_iou=False, focaler_iou=False)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,
                mpdiou_hw=None):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        if self.use_wiseiou:
            wiou = self.wiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], ret_iou=False, ratio=0.7, d=0.0,
                                  u=0.95).unsqueeze(-1)
            # wiou = self.wiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], ret_iou=False, ratio=0.7, d=0.0, u=0.95, **{'scale':0.0}).unsqueeze(-1) # Wise-ShapeIoU,Wise-Inner-ShapeIoU,Wise-Focaler-ShapeIoU
            # wiou = self.wiou_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask], ret_iou=False, ratio=0.7, d=0.0, u=0.95, **{'mpdiou_hw':mpdiou_hw[fg_mask]}).unsqueeze(-1) # Wise-MPDIoU,Wise-Inner-MPDIoU,Wise-Focaler-MPDIoU
            loss_iou = (wiou * weight).sum() / target_scores_sum
        else:
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            # iou = bbox_inner_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True, ratio=0.7)
            # iou = bbox_mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, mpdiou_hw=mpdiou_hw[fg_mask])
            # iou = bbox_inner_mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, mpdiou_hw=mpdiou_hw[fg_mask], ratio=0.7)
            # iou = bbox_focaler_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True, d=0.0, u=0.95)
            # iou = bbox_focaler_mpdiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, mpdiou_hw=mpdiou_hw[fg_mask], d=0.0, u=0.95)
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.nwd_loss:
            nwd = wasserstein_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask])
            nwd_loss = ((1.0 - nwd) * weight).sum() / target_scores_sum
            loss_iou = self.iou_ratio * loss_iou + (1 - self.iou_ratio) * nwd_loss

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl
#
# class BboxLoss(nn.Module):
#     """Criterion class for computing training losses during training."""
#
#     def __init__(self, reg_max=16):
#         """Initialize the BboxLoss module with regularization maximum and DFL settings."""
#         super().__init__()
#         self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
#
#     def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
#         """IoU loss."""
#         weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
#         iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
#         loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
#
#         # DFL loss
#         if self.dfl_loss:
#             target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
#             loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
#             loss_dfl = loss_dfl.sum() / target_scores_sum
#         else:
#             loss_dfl = torch.tensor(0.0).to(pred_dist.device)
#
#         return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        # self.bce = EMASlideLoss(nn.BCEWithLogitsLoss(reduction='none'))  # Exponential Moving Average Slide Loss
        # self.bce = SlideLoss(nn.BCEWithLogitsLoss(reduction='none')) # Slide Loss
        # self.bce = FocalLoss_YOLO(alpha=0.25, gamma=1.5) # FocalLoss
        # self.bce = VarifocalLoss_YOLO(alpha=0.75, gamma=2.0) # VarifocalLoss
        # self.bce = QualityfocalLoss_YOLO(beta=2.0) # QualityfocalLoss
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        if hasattr(m, 'dfl_aux'):
            self.assigner_aux = TaskAlignedAssigner(topk=13, num_classes=self.nc, alpha=0.5, beta=6.0)
            self.aux_loss_ratio = 0.25
        # self.assigner = ATSSAssigner(9, num_classes=self.nc)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        # ATSS use
        self.grid_cell_offset = 0.5
        self.fpn_strides = list(self.stride.detach().cpu().numpy())
        self.grid_cell_size = 5.0

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            # pred_dist: (b, a, c)  where c = 4 * reg_max
            # pred_dist = pred_dist.float()  # 👈 关键！
            # self.proj = self.proj.float()  # 确保 proj 也是 float32
            # print("pred_dist dtype:", pred_dist.dtype)
            # print("pred_dist min:", pred_dist.min().item())
            # print("pred_dist max:", pred_dist.max().item())
            # print("pred_dist has nan:", torch.isnan(pred_dist).any().item())

            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        if hasattr(self, 'assigner_aux'):
            loss, batch_size = self.compute_loss_aux(preds, batch)
        else:
            loss, batch_size = self.compute_loss(preds, batch)
        return loss.sum() * batch_size, loss.detach() #loss求和

    def compute_loss(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        # print("len(preds):", len(preds))# 4
        # print("preds[3][0].shape:",preds[3][0].shape) #[B,2]
        # print("preds[3][1].shape:",preds[3][1].shape)
        # print("preds[3][2].shape:",preds[3][2].shape)
        # print("len(batch):", len(batch))
        # print(preds)
        # print(batch)
        # print("batch.shape:",batch.shape)
        preds = preds[1] if isinstance(preds, tuple) else preds ##验证的时候返回的是tuple
        # preds, txty_preds, rgbxy_preds = preds[:3], preds[3:4], preds[4:]
        # self.use_offset = self.hyp.use_offset
        # self.use_rgbcenter = self.hyp.rgbcenter
        # self.use_offset = False
        # self.use_rgbcenter = False

        # if len(preds)==6:
        #     preds, txty_preds = preds[:3], preds[3:]
        #     self.use_offset = True
        # if len(preds)==9:
        #     ppreds, txty_preds, rgbxy_preds = preds[:3], preds[3:6], preds[6:]
        #     self.use_rgbcenter = True

        if len(preds)==6:
            preds, centerpoints, uav_exist, txty_preds = preds[:3], preds[3], preds[4], preds[5]
        else:
            centerpoints, uav_exist, txty_preds = None, None, None

        batch['offset'] = torch.from_numpy(np.array(batch['offset'], dtype=np.float32)).to('cuda')
        batch['area'] = torch.from_numpy(np.array(batch['area'], dtype=np.float32)).to('cuda')
        batch['center_point'] = torch.from_numpy(np.array(batch['center_point'], dtype=np.float32)).to('cuda')
        batch['cls_rgb'] = torch.from_numpy(np.array(batch['cls_rgb'], dtype=np.float32)).to('cuda')+1 #+1使得-1，0转为0，1的二分类，也刚好与存在无人机的概率相同 
        # print("boxes_rgb：", batch['boxes_rgb'])
        batch['boxes_rgb'] = torch.from_numpy(np.array(batch['boxes_rgb'], dtype=np.float32)).to('cuda')
        batch['boxes_ir'] = torch.from_numpy(np.array(batch['boxes_ir'], dtype=np.float32)).to('cuda')


        # loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        loss = torch.zeros(6, device=self.device)  # box, cls, dfl, offset, rgbcenter, uav_exist
        feats = preds[1] if isinstance(preds, tuple) else preds
        feats = feats[:self.stride.size(0)]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        # targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]]) #xywh转为xyxy
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)#只有xyxy加和>0的才是有效目标

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        # ATSS
        if isinstance(self.assigner, ATSSAssigner):
            anchors, _, n_anchors_list, _ = \
                generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset,
                                 device=feats[0].device)
            target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(anchors, n_anchors_list, gt_labels,
                                                                                    gt_bboxes, mask_gt,
                                                                                    pred_bboxes.detach() * stride_tensor)
        # TAL
        else:
            target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
                pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        if isinstance(self.bce, (nn.BCEWithLogitsLoss, FocalLoss_YOLO)):
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        elif isinstance(self.bce, VarifocalLoss_YOLO):
            if fg_mask.sum():
                pos_ious = bbox_iou(pred_bboxes, target_bboxes / stride_tensor, xywh=False).clamp(min=1e-6).detach()
                # 10.0x Faster than torch.one_hot
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                              dtype=torch.int64,
                                              device=target_labels.device)  # (b, h*w, 80)
                cls_iou_targets.scatter_(2, target_labels.unsqueeze(-1), 1)
                cls_iou_targets = pos_ious * cls_iou_targets
                fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, 80)
                cls_iou_targets = torch.where(fg_scores_mask > 0, cls_iou_targets, 0)
            else:
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                              dtype=torch.int64,
                                              device=target_labels.device)  # (b, h*w, 80)
            loss[1] = self.bce(pred_scores, cls_iou_targets.to(dtype)).sum() / max(fg_mask.sum(), 1)  # BCE
        elif isinstance(self.bce, QualityfocalLoss_YOLO):
            if fg_mask.sum():
                pos_ious = bbox_iou(pred_bboxes, target_bboxes / stride_tensor, xywh=False).clamp(min=1e-6).detach()
                # 10.0x Faster than torch.one_hot
                targets_onehot = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                             dtype=torch.int64,
                                             device=target_labels.device)  # (b, h*w, 80)
                targets_onehot.scatter_(2, target_labels.unsqueeze(-1), 1)
                cls_iou_targets = pos_ious * targets_onehot
                fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)  # (b, h*w, 80)
                targets_onehot_pos = torch.where(fg_scores_mask > 0, targets_onehot, 0)
                cls_iou_targets = torch.where(fg_scores_mask > 0, cls_iou_targets, 0)
            else:
                cls_iou_targets = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                              dtype=torch.int64,
                                              device=target_labels.device)  # (b, h*w, 80)
                targets_onehot_pos = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.nc),
                                                 dtype=torch.int64,
                                                 device=target_labels.device)  # (b, h*w, 80)
            loss[1] = self.bce(pred_scores, cls_iou_targets.to(dtype), targets_onehot_pos.to(torch.bool)).sum() / max(
                fg_mask.sum(), 1)

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask,
                                              ((imgsz[0] ** 2 + imgsz[1] ** 2) / torch.square(stride_tensor)).repeat(1,
                                                                                                                     batch_size).transpose(
                                                  1, 0))

        if isinstance(self.bce, (EMASlideLoss, SlideLoss)):
            if fg_mask.sum():
                auto_iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True).mean()
            else:
                auto_iou = -1
            loss[1] = self.bce(pred_scores, target_scores.to(dtype), auto_iou).sum() / target_scores_sum  # BCE

        if txty_preds!=None:
            # loss[3] = self.compute_offset_loss(txty_preds, batch['offset'][:,:2], batch['area'])
            # loss[3] = torch.zeros(loss[2].shape)
            loss[3] = self.affine_xy_center_loss(txty_preds, batch['boxes_rgb'], batch['boxes_ir'], True, True)
            # loss[3] = self.affine_xy_center_loss(txty_preds, batch['boxes_rgb'], batch['boxes_ir'], True, True)
            # loss[3] = self.affine_box_corner_loss(txty_preds, batch['boxes_rgb'], batch['boxes_ir'])
            if loss[3]==None:
                loss[3] = torch.zeros(loss[2].shape)
        else:
            loss[3] = torch.zeros(loss[2].shape)

        if centerpoints!=None:
            loss[4] = self.compute_rgbcenter_loss(centerpoints, batch['center_point'])
        else:
            loss[4] = torch.zeros(loss[2].shape)

        if uav_exist!=None:
            # loss[5] = self.compute_uavExist_loss(uav_exist, batch['cls_rgb'])
            # loss[5] = self.bce(uav_exist, batch['cls_rgb']).mean()
            loss[5] = FocalLossProb.forward(
                pred=uav_exist,
                label=batch['cls_rgb'],
                gamma=2.0,
                alpha=0.1
            )
            # print("uav_exist:", uav_exist)
            # print("batch['cls_rgb']:", batch['cls_rgb'])
        else:
            loss[5] = torch.zeros(loss[2].shape)
        # print("self.hyp.offset:", self.hyp.offset)  #0.1
        # print("loss[3]33333333333344444444", loss[3])
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.offset  # offset gain
        loss[4] *= self.hyp.rgbcenter  # rgbcenter gain
        loss[5] *= self.hyp.uav_exist  # uav_exist gain
        # print("Loss[3]3333333333:", loss[3])
        return loss, batch_size

    def compute_loss_aux(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats_all = preds[1] if isinstance(preds, tuple) else preds
        if len(feats_all) == self.stride.size(0):
            return self.compute_loss(preds, batch)
        feats, feats_aux = feats_all[:self.stride.size(0)], feats_all[self.stride.size(0):]

        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)
        pred_distri_aux, pred_scores_aux = torch.cat([xi.view(feats_aux[0].shape[0], self.no, -1) for xi in feats_aux],
                                                     2).split((self.reg_max * 4, self.nc), 1)

        pred_scores, pred_distri = pred_scores.permute(0, 2, 1).contiguous(), pred_distri.permute(0, 2, 1).contiguous()
        pred_scores_aux, pred_distri_aux = pred_scores_aux.permute(0, 2, 1).contiguous(), pred_distri_aux.permute(0, 2,
                                                                                                                  1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_bboxes_aux = self.bbox_decode(anchor_points, pred_distri_aux)  # xyxy, (b, h*w, 4)

        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(pred_scores.detach().sigmoid(), (
                    pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                                                                                anchor_points * stride_tensor,
                                                                                gt_labels, gt_bboxes, mask_gt)
        target_labels_aux, target_bboxes_aux, target_scores_aux, fg_mask_aux, _ = self.assigner_aux(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)
        target_scores_sum_aux = max(target_scores_aux.sum(), 1)

        # cls loss
        if isinstance(self.bce, nn.BCEWithLogitsLoss):
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
            loss[1] += self.bce(pred_scores_aux,
                                target_scores_aux.to(dtype)).sum() / target_scores_sum_aux * self.aux_loss_ratio  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            target_bboxes_aux /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask,
                                              ((imgsz[0] ** 2 + imgsz[1] ** 2) / torch.square(stride_tensor)).repeat(1,
                                                                                                                     batch_size).transpose(
                                                  1, 0))
            aux_loss_0, aux_loss_2 = self.bbox_loss(pred_distri_aux, pred_bboxes_aux, anchor_points, target_bboxes_aux,
                                                    target_scores_aux,
                                                    target_scores_sum_aux, fg_mask_aux, (
                                                                (imgsz[0] ** 2 + imgsz[1] ** 2) / torch.square(
                                                            stride_tensor)).repeat(1, batch_size).transpose(1, 0))

            loss[0] += aux_loss_0 * self.aux_loss_ratio
            loss[2] += aux_loss_2 * self.aux_loss_ratio

        if isinstance(self.bce, (EMASlideLoss, SlideLoss)):
            auto_iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True).mean()
            loss[1] = self.bce(pred_scores, target_scores.to(dtype), auto_iou).sum() / target_scores_sum  # BCE
            loss[1] += self.bce(pred_scores_aux, target_scores_aux.to(dtype),
                                -1).sum() / target_scores_sum_aux * self.aux_loss_ratio  # BCE

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        # return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
        return loss, batch_size

    def compute_offset_loss(self, txtys, targets, area_ir):
        """
        计算约束后的偏移损失：
        - txtys包含3层预测偏移，分别与targets计算损失后累加
        - boxes_ir全0的样本（背景图片）不计入损失
        """
        total_loss_offset = 0.0  # 总损失（累加3个预测结果的损失）

        # ===================== 1. 解析并规整输入 =====================
        # 拆解txtys的嵌套结构：[[t1,t2,t3]] → [t1,t2,t3]
        txtys_flat = txtys[0] if (isinstance(txtys, list) and len(txtys) == 1) else txtys
        # 确保targets是Tensor且固定设备（以targets设备为基准）
        device = targets.device
        offset_gt = targets.reshape(-1, 2).to(device) * 2  # [16,2],坐标系为0，1，而grid为-1，1，需要乘以2
        area_ir = area_ir.reshape(-1, 1).to(device)  # [16,1]

        # ===================== 2. 生成有效样本掩码（仅计算一次） =====================
        # 有效样本：boxes_ir每行前4列不全为0
        valid_mask = ~(area_ir[:, :] == 0).all(dim=1)  # [16,]，True=有效样本
        valid_num = valid_mask.sum().clamp(min=1)  # 有效样本数，避免除0
        # 预计算有效样本的面积均值（仅计算一次，复用）
        area_mean = (area_ir.sum() / valid_num).clamp(min=1e-6)  # 有效面积均值

        # ===================== 3. 遍历3个预测结果，分别计算损失 =====================
        for idx, pred_tensor in enumerate(txtys_flat):
            # 规整单个预测Tensor：设备对齐+维度为[16,2]
            offset_pred = pred_tensor.to(device).reshape(-1, 2)  # [16,2]
            
            # -------------------- 计算单个预测结果的损失 --------------------
            # 仅计算有效样本的tx/ty损失（无效样本置0）
            loss_tx = torch.abs(offset_pred[:, 0] - offset_gt[:, 0]) * valid_mask.float()
            loss_ty = torch.abs(offset_pred[:, 1] - offset_gt[:, 1]) * valid_mask.float()
            
            # 单个预测结果的平均损失（仅有效样本）
            single_loss = (loss_tx + loss_ty).sum() / valid_num
            # 归一化（除以面积均值）
            single_loss /= area_mean
            
            # 累加至总损失
            total_loss_offset += single_loss
            # 可选：打印每个预测结果的损失（仅rank0打印）
            # if (not hasattr(self, 'rank')) or (self.rank == 0):
            #     print(f"第{idx+1}个预测结果的offset损失：{single_loss.item():.6f}")

        return total_loss_offset

    def compute_rgbcenter_loss(self, preds, targets):
        total_loss = 0.0  # 总损失（累加3个预测结果的损失）

        # ===================== 1. 解析并规整输入 =====================
        # 拆解txtys的嵌套结构：[[t1,t2,t3]] → [t1,t2,t3]
        preds_flat = preds[0] if (isinstance(preds, list) and len(preds) == 1) else preds
        device = targets.device
        targets = targets.reshape(-1, 2).to(device)  # [16,2]

        # ===================== 3. 遍历3个预测结果，分别计算损失 =====================
        for idx, pred_tensor in enumerate(preds_flat):
            # 规整单个预测Tensor：设备对齐+维度为[16,2]
            if not(targets[idx,0]==0 and targets[idx,1]==0):
                rgbcenter_pred = pred_tensor.to(device).reshape(-1, 2)  # [16,2]
                
                loss_x = torch.abs(rgbcenter_pred[:, 0] - targets[:, 0]) 
                loss_y = torch.abs(rgbcenter_pred[:, 1] - targets[:, 1]) 
                
                single_loss = (loss_x + loss_y).sum()
                total_loss += single_loss
            # 可选：打印每个预测结果的损失（仅rank0打印）
            # if (not hasattr(self, 'rank')) or (self.rank == 0):
            #     print(f"第{idx+1}个预测结果的offset损失：{single_loss.item():.6f}")

        return total_loss

    def compute_uavExist_loss(self, preds, targets):
        total_loss = 0.0  # 总损失（累加3个预测结果的损失）

        # print("target.shape:",targets.shape)
        # ===================== 1. 解析并规整输入 =====================
        # 拆解txtys的嵌套结构：[[t1,t2,t3]] → [t1,t2,t3]
        preds_flat = preds[0] if (isinstance(preds, list) and len(preds) == 1) else preds
        device = targets.device
        targets = targets.reshape(-1, 1).to(device)  # [16,2]
        # print("target.shape:",targets.shape)

        # ===================== 3. 遍历3个预测结果，分别计算损失 =====================
        for idx, pred_tensor in enumerate(preds_flat):
            uav_exist = pred_tensor.to(device).reshape(-1, 1)  # [16,2]
            # print("uav_exist.shape:",uav_exist.shape)
            single_loss = self.bce(uav_exist, targets).mean()
            total_loss += single_loss

        return total_loss




    def affine_xy_center_loss(self, pred_affine_params, rgb_boxes_xywh, ir_boxes_xywh,
                            use_area_scale: bool = False, use_smooth_l1: bool = False):
        B = pred_affine_params.shape[0]
        device = pred_affine_params.device

        # 步骤1：提取中心坐标 + 计算IR框面积（仅当use_area_scale=True时使用）
        rgb_x_c = rgb_boxes_xywh[:, 0]  # [B]
        rgb_y_c = rgb_boxes_xywh[:, 1]  # [B]
        ir_x_c = ir_boxes_xywh[:, 0]    # [B]
        ir_y_c = ir_boxes_xywh[:, 1]    # [B]
        ir_w = ir_boxes_xywh[:, 2]
        ir_h = ir_boxes_xywh[:, 3]
        ir_area = ir_w * ir_h + 1e-8    # [B]，+1e-8避免除以0

        # 步骤2：生成有效样本掩码
        rgb_center_zero = (rgb_x_c == 0) & (rgb_y_c == 0)
        ir_center_zero = (ir_x_c == 0) & (ir_y_c == 0)
        invalid_mask = rgb_center_zero | ir_center_zero
        valid_mask = ~invalid_mask
        valid_num = valid_mask.sum().float()

        # 无有效样本返回0损失
        if valid_num < 1e-6:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 步骤3：构造仿射矩阵 + 变换RGB中心
        a = pred_affine_params[:, 0]
        b = pred_affine_params[:, 1]
        c = pred_affine_params[:, 2]
        d = pred_affine_params[:, 3]
        tx = pred_affine_params[:, 4]
        ty = pred_affine_params[:, 5]

        # 批量仿射矩阵 [B, 2, 3]
        M = torch.stack([
            torch.stack([a, b, tx], dim=1),
            torch.stack([c, d, ty], dim=1)
        ], dim=1)

        # RGB中心齐次坐标变换
        rgb_center = torch.stack([rgb_x_c, rgb_y_c], dim=1)  # [B, 2]
        rgb_center_homo = torch.cat([
            rgb_center,
            torch.ones([B, 1], device=device, dtype=rgb_center.dtype)
        ], dim=1)  # [B, 3]
        rgb_center_homo = rgb_center_homo.unsqueeze(-1)  # [B, 3, 1]
        rgb_center_transformed_homo = torch.bmm(M, rgb_center_homo)  # [B, 2, 1]
        rgb_center_transformed = rgb_center_transformed_homo.squeeze(-1)  # [B, 2]

        pred_x_c = rgb_center_transformed[:, 0]
        pred_y_c = rgb_center_transformed[:, 1]

        # 步骤4：筛选有效样本
        valid_pred_x = pred_x_c[valid_mask]
        valid_pred_y = pred_y_c[valid_mask]
        valid_ir_x = ir_x_c[valid_mask]
        valid_ir_y = ir_y_c[valid_mask]
        valid_ir_area = ir_area[valid_mask]

        # 步骤5：计算偏移损失（可选Smooth L1 / 普通L1）
        if use_smooth_l1:
            # Smooth L1 Loss：更鲁棒，对异常值不敏感
            x_loss = F.smooth_l1_loss(valid_pred_x, valid_ir_x, reduction='none')
            y_loss = F.smooth_l1_loss(valid_pred_y, valid_ir_y, reduction='none')
        else:
            # 普通L1 Loss：计算简单，偏移量直观
            x_loss = torch.abs(valid_pred_x - valid_ir_x)
            y_loss = torch.abs(valid_pred_y - valid_ir_y)

        # 步骤6：可选：偏移损失 ÷ IR框面积（放大偏移）
        if use_area_scale:
            x_loss = x_loss / valid_ir_area
            y_loss = y_loss / valid_ir_area

        # 单个样本损失（x+y偏移）
        single_sample_loss = x_loss + y_loss  # [有效B]

        # 步骤7：最终损失：有效样本的普通均值（mean）
        center_loss = single_sample_loss.mean()

        return center_loss

    def affine_box_corner_loss(self, pred_affine_params, rgb_boxes_xywh, ir_boxes_xywh,
                            use_area_scale: bool = False, use_smooth_l1: bool = False):

        B = pred_affine_params.shape[0]
        device = pred_affine_params.device

        # 步骤1：XYWH转[x1,y1,x2,y2]（左上+右下） + 计算IR框面积
        # RGB框转换
        rgb_x_c = rgb_boxes_xywh[:, 0]
        rgb_y_c = rgb_boxes_xywh[:, 1]
        rgb_w = rgb_boxes_xywh[:, 2]
        rgb_h = rgb_boxes_xywh[:, 3]
        rgb_x1 = rgb_x_c - (rgb_w / 2.0)
        rgb_y1 = rgb_y_c - (rgb_h / 2.0)
        rgb_x2 = rgb_x_c + (rgb_w / 2.0)
        rgb_y2 = rgb_y_c + (rgb_h / 2.0)

        # IR框转换 + 面积计算
        ir_x_c = ir_boxes_xywh[:, 0]
        ir_y_c = ir_boxes_xywh[:, 1]
        ir_w = ir_boxes_xywh[:, 2]
        ir_h = ir_boxes_xywh[:, 3]
        ir_x1 = ir_x_c - (ir_w / 2.0)
        ir_y1 = ir_y_c - (ir_h / 2.0)
        ir_x2 = ir_x_c + (ir_w / 2.0)
        ir_y2 = ir_y_c + (ir_h / 2.0)
        ir_area = ir_w * ir_h + 1e-8  # [B]，+1e-8避免除以0

        # 步骤2：提取两角坐标 + 生成有效样本掩码
        # RGB两角 [B, 2, 2]：[批量, 角数量(2), 坐标(x,y)]
        rgb_corners = torch.stack([
            torch.stack([rgb_x1, rgb_y1], dim=1),  # 左上
            torch.stack([rgb_x2, rgb_y2], dim=1)   # 右下
        ], dim=1)

        # IR两角 [B, 2, 2]
        ir_corners = torch.stack([
            torch.stack([ir_x1, ir_y1], dim=1),  # 左上
            torch.stack([ir_x2, ir_y2], dim=1)   # 右下
        ], dim=1)

        # 有效样本掩码
        rgb_center_zero = (rgb_x_c == 0) & (rgb_y_c == 0)
        ir_center_zero = (ir_x_c == 0) & (ir_y_c == 0)
        invalid_mask = rgb_center_zero | ir_center_zero
        valid_mask = ~invalid_mask
        valid_num = valid_mask.sum().float()

        # 无有效样本返回0损失
        if valid_num < 1e-6:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 步骤3：构造仿射矩阵 + 变换RGB两角
        a = pred_affine_params[:, 0]
        b = pred_affine_params[:, 1]
        c = pred_affine_params[:, 2]
        d = pred_affine_params[:, 3]
        tx = pred_affine_params[:, 4]
        ty = pred_affine_params[:, 5]

        # 批量仿射矩阵 [B, 2, 3]
        M = torch.stack([
            torch.stack([a, b, tx], dim=1),
            torch.stack([c, d, ty], dim=1)
        ], dim=1)

        # RGB两角齐次坐标变换
        rgb_corners_homo = torch.cat([
            rgb_corners,
            torch.ones([B, 2, 1], device=device, dtype=rgb_corners.dtype)
        ], dim=2)  # [B, 2, 3]
        rgb_corners_homo = rgb_corners_homo.transpose(1, 2)  # [B, 3, 2]
        rgb_corners_transformed = torch.bmm(M, rgb_corners_homo)  # [B, 2, 2]

        # 步骤4：计算偏移损失（可选面积缩放 + Smooth L1）
        valid_loss_list = []
        for b_idx in range(B):
            if not valid_mask[b_idx]:
                continue  # 跳过无效样本

            # 单个样本的变换后RGB两角和IR两角
            pred_corners = rgb_corners_transformed[b_idx]  # [2, 2]
            target_corners = ir_corners[b_idx]              # [2, 2]

            # 计算两角x/y偏移（可选Smooth L1 / 普通L1）
            if use_smooth_l1:
                x_loss = F.smooth_l1_loss(pred_corners[:, 0], target_corners[:, 0], reduction='none')
                y_loss = F.smooth_l1_loss(pred_corners[:, 1], target_corners[:, 1], reduction='none')
            else:
                x_loss = torch.abs(pred_corners[:, 0] - target_corners[:, 0])
                y_loss = torch.abs(pred_corners[:, 1] - target_corners[:, 1])

            # 可选：偏移损失 ÷ 该样本的IR框面积（放大偏移）
            if use_area_scale:
                x_loss = x_loss / ir_area[b_idx]
                y_loss = y_loss / ir_area[b_idx]

            # 单个样本的两角总损失（平均后存入列表）
            single_corner_loss = (x_loss.sum() + y_loss.sum()) / 2.0
            valid_loss_list.append(single_corner_loss)

        # 步骤5：转换为张量 + 最终普通均值（mean）
        valid_loss_tensor = torch.stack(valid_loss_list)  # [有效B]
        corner_loss = valid_loss_tensor.mean()

        return corner_loss

# class v8DetectionLoss:
#     """Criterion class for computing training losses."""
#
#     def __init__(self, model, tal_topk=10):  # model must be de-paralleled
#         """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
#         device = next(model.parameters()).device  # get model device
#         h = model.args  # hyperparameters
#
#         m = model.model[-1]  # Detect() module
#         self.bce = nn.BCEWithLogitsLoss(reduction="none")
#         self.hyp = h
#         self.stride = m.stride  # model strides
#         self.nc = m.nc  # number of classes
#         self.no = m.nc + m.reg_max * 4
#         self.reg_max = m.reg_max
#         self.device = device
#
#         self.use_dfl = m.reg_max > 1
#
#         self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
#         self.bbox_loss = BboxLoss(m.reg_max).to(device)
#         self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
#
#     def preprocess(self, targets, batch_size, scale_tensor):
#         """Preprocesses the target counts and matches with the input batch size to output a tensor."""
#         nl, ne = targets.shape
#         if nl == 0:
#             out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
#         else:
#             i = targets[:, 0]  # image index
#             _, counts = i.unique(return_counts=True)
#             counts = counts.to(dtype=torch.int32)
#             out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
#             for j in range(batch_size):
#                 matches = i == j
#                 if n := matches.sum():
#                     out[j, :n] = targets[matches, 1:]
#             out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
#         return out
#
#     def bbox_decode(self, anchor_points, pred_dist):
#         """Decode predicted object bounding box coordinates from anchor points and distribution."""
#         if self.use_dfl:
#             b, a, c = pred_dist.shape  # batch, anchors, channels
#             pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
#             # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
#             # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
#         return dist2bbox(pred_dist, anchor_points, xywh=False)
#
#     def __call__(self, preds, batch):
#         """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
#         loss = torch.zeros(3, device=self.device)  # box, cls, dfl
#         feats = preds[1] if isinstance(preds, tuple) else preds
#         pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
#             (self.reg_max * 4, self.nc), 1
#         )
#
#         pred_scores = pred_scores.permute(0, 2, 1).contiguous()
#         pred_distri = pred_distri.permute(0, 2, 1).contiguous()
#
#         dtype = pred_scores.dtype
#         batch_size = pred_scores.shape[0]
#         imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
#         anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
#
#         # Targets
#         targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
#         targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
#         gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
#         mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
#
#         # Pboxes
#         pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
#         # dfl_conf = pred_distri.view(batch_size, -1, 4, self.reg_max).detach().softmax(-1)
#         # dfl_conf = (dfl_conf.amax(-1).mean(-1) + dfl_conf.amax(-1).amin(-1)) / 2
#
#         _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
#             # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
#             pred_scores.detach().sigmoid(),
#             (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
#             anchor_points * stride_tensor,
#             gt_labels,
#             gt_bboxes,
#             mask_gt,
#         )
#
#         target_scores_sum = max(target_scores.sum(), 1)
#
#         # Cls loss
#         # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
#         loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
#
#         # Bbox loss
#         if fg_mask.sum():
#             target_bboxes /= stride_tensor
#             loss[0], loss[2] = self.bbox_loss(
#                 pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
#             )
#
#         loss[0] *= self.hyp.box  # box gain
#         loss[1] *= self.hyp.cls  # cls gain
#         loss[2] *= self.hyp.dfl  # dfl gain
#
#         return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
#

class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model):
        """Initializes v8OBBLoss with model, assigner, and rotated bbox loss; note model must be de-paralleled."""
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo11n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)


class E2EDetectLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]
