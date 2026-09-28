"""Segmentation loss functions, PyTorch port of dl_tools.keras_custom_loss.

All coefficient/loss functions take (y_true, y_pred) tensors of the same
shape (matching the Keras convention used throughout the original file),
except where the original explicitly worked on logits (tversky_loss,
lovasz_*), which is preserved here.
"""
import numpy as np
import torch
import torch.nn.functional as F

SMOOTH = 1.0


# --------------------------------------------------------------------------
# Dice / Jaccard (soft, differentiable)
# --------------------------------------------------------------------------

def dice1_coef(y_true, y_pred, smooth=SMOOTH):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


def dice1_loss(y_true, y_pred, smooth=SMOOTH):
    return 1 - dice1_coef(y_true, y_pred, smooth)


def dice2_coef(y_true, y_pred, smooth=SMOOTH):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        torch.sum(y_true_f * y_true_f) + torch.sum(y_pred_f * y_pred_f) + smooth)


def dice2_loss(y_true, y_pred, smooth=SMOOTH):
    return 1 - dice2_coef(y_true, y_pred, smooth)


def jaccard1_coef(y_true, y_pred, smooth=SMOOTH):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def jaccard1_loss(y_true, y_pred, smooth=SMOOTH):
    return 1 - jaccard1_coef(y_true, y_pred, smooth)


def jaccard2_coef(y_true, y_pred, smooth=SMOOTH):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f * y_true_f) + torch.sum(y_pred_f * y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def jaccard2_loss(y_true, y_pred, smooth=SMOOTH):
    return 1 - jaccard2_coef(y_true, y_pred, smooth)


# --------------------------------------------------------------------------
# Focal loss (binary and categorical)
# https://arxiv.org/pdf/1708.02002.pdf
# --------------------------------------------------------------------------

def binary_focal_loss(gamma=2., alpha=.25):
    """Returns a loss_fn(y_true, y_pred) with y_pred already a sigmoid output."""

    def loss_fn(y_true, y_pred):
        y_true = y_true.float()
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
        p_t = torch.where(y_true == 1, y_pred, 1 - y_pred)
        alpha_t = torch.where(y_true == 1, torch.full_like(y_true, alpha),
                               torch.full_like(y_true, 1 - alpha))
        cross_entropy = -torch.log(p_t)
        weight = alpha_t * (1 - p_t) ** gamma
        loss = weight * cross_entropy
        # sum over the channel dim, mean over the rest (mirrors K.sum(axis=1))
        return loss.sum(dim=1).mean()

    return loss_fn


def categorical_focal_loss(alpha, gamma=2.):
    """alpha: per-class weight, broadcastable to (num_classes, H, W) or scalar-like."""
    alpha_t = torch.as_tensor(np.array(alpha, dtype=np.float32))

    def loss_fn(y_true, y_pred):
        eps = 1e-7
        y_pred = torch.clamp(y_pred, eps, 1. - eps)
        cross_entropy = -y_true * torch.log(y_pred)
        a = alpha_t.to(y_pred.device)
        loss = a * (1 - y_pred) ** gamma * cross_entropy
        return loss.sum(dim=1).mean()

    return loss_fn


def tversky_loss(beta):
    """Operates on logits (applies sigmoid internally), as in the original."""

    def loss_fn(y_true, logits):
        y_true = y_true.float()
        y_pred = torch.sigmoid(logits)
        numerator = y_true * y_pred
        denominator = (y_true * y_pred + beta * (1 - y_true) * y_pred
                        + (1 - beta) * y_true * (1 - y_pred))
        return 1 - torch.sum(numerator) / torch.sum(denominator)

    return loss_fn


# --------------------------------------------------------------------------
# Lovasz hinge / softmax
# adapted from https://github.com/bermanmaxim/LovaszSoftmax
# --------------------------------------------------------------------------

def lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension w.r.t sorted errors (Alg. 1)."""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1. - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / union
    if jaccard.numel() > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def flatten_binary_scores(scores, labels, ignore=None):
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    return scores[valid], labels[valid]


def lovasz_hinge_flat(logits, labels):
    if labels.numel() == 0:
        return logits.sum() * 0.
    labelsf = labels.float()
    signs = 2. * labelsf - 1.
    errors = 1. - logits * signs.detach()
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labelsf[perm]
    grad = lovasz_grad(gt_sorted).detach()
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """logits, labels: (B, H, W) tensors."""
    if per_image:
        losses = []
        for log, lab in zip(logits, labels):
            log_f, lab_f = flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)
            losses.append(lovasz_hinge_flat(log_f, lab_f))
        return torch.stack(losses).mean()
    else:
        log_f, lab_f = flatten_binary_scores(logits, labels, ignore)
        return lovasz_hinge_flat(log_f, lab_f)


def lovasz_softmax(y_true, y_pred_logits):
    """Kept with the (y_true, y_pred) argument order used elsewhere in this
    file, unlike the upstream lovasz_hinge(logits, labels) order."""
    return lovasz_hinge(y_pred_logits, y_true)
