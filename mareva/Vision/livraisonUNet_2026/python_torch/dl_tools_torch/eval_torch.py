"""Segmentation evaluation functions.

`jaccard` / `jaccard_curve` are unchanged (pure numpy). The confusion-matrix
based metrics from dl_tools.tools (myEval / evaluate / computeFMeanFromCM,
which relied on tf.keras.metrics.MeanIoU) are reimplemented here with plain
numpy so no TensorFlow dependency remains.
"""
import numpy as np
import matplotlib.pyplot as plt

from dl_tools_torch.dataset_torch import to_one_hot  # re-exported for convenience


def jaccard(im1, im2):
    """Jaccard index between binary/positive-integer images im1 and im2."""
    union_vol = np.sum(np.maximum(im1, im2))
    if union_vol <= 0:
        raise ValueError("Images are empty (or contain negative values)")
    return np.sum(np.minimum(im1, im2)) / union_vol


def jaccard_curve(im_grey, im_bin):
    """Jaccard index for every threshold in [0, 255]."""
    values = []
    for grey in range(256):
        im_thresh = im_grey > grey
        values.append(jaccard(im_bin, im_thresh))
    return values


def confusion_matrix(y_true, y_pred, num_classes):
    """Confusion matrix, transposed like the original tools.py (rows =
    predicted class, cols = ground-truth class)."""
    y_true = y_true.reshape(-1).astype(np.int64)
    y_pred = y_pred.reshape(-1).astype(np.int64)
    idx = y_true * num_classes + y_pred
    cm = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm.T


def compute_f_mean_from_cm(conf_mat):
    """Per-class precision, recall, F-measure and IoU from a confusion matrix."""
    shape = conf_mat.shape
    sum_col = np.sum(conf_mat, axis=0)
    sum_lines = np.sum(conf_mat, axis=1)
    precision = np.zeros(shape[0], dtype=np.float32)
    recall = np.zeros(shape[0], dtype=np.float32)
    fmean = np.zeros(shape[0], dtype=np.float32)
    iou = np.zeros(shape[0], dtype=np.float32)
    for i in range(shape[0]):
        sum_col_line = sum_col[i] + sum_lines[i] - conf_mat[i, i]
        precision[i] = conf_mat[i, i] / sum_col[i] if sum_col[i] > 0 else 0
        recall[i] = conf_mat[i, i] / sum_lines[i] if sum_lines[i] > 0 else 0
        iou[i] = conf_mat[i, i] / sum_col_line if sum_col_line > 0 else 0
        if precision[i] > 0 and recall[i] > 0:
            fmean[i] = 2.0 * precision[i] * recall[i] / (precision[i] + recall[i])
    return precision, recall, fmean, iou


def mean_iou(conf_mat):
    _, _, _, iou = compute_f_mean_from_cm(conf_mat)
    return float(np.mean(iou))


def plot_confusion_matrix(conf_mat, file_name, disp_labs, show=False, plot_scale=0.3):
    default_figsize = plt.rcParams['figure.figsize']
    if show:
        plt.rcParams['figure.figsize'] = [v * plot_scale for v in default_figsize]

    fig, ax = plt.subplots()
    ax.matshow(conf_mat, cmap=plt.cm.Blues)
    xsize, ysize = conf_mat.shape
    for i in range(xsize):
        for j in range(ysize):
            ax.text(i, j, str(conf_mat[j, i]), va='center', ha='center')
    ax.set_xticklabels(disp_labs)
    ax.set_yticklabels(disp_labs)
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.1)
    if show:
        plt.show()
    plt.clf()

    if show:
        plt.rcParams['figure.figsize'] = default_figsize


def evaluate(y_pred, y_gt, num_classes, dir_res, base_name, show=False):
    """Port of dl_tools.tools.evaluate: y_pred/y_gt contain integer labels
    (not one-hot / probabilities)."""
    cm = confusion_matrix(y_gt, y_pred, num_classes)
    disp_labs = [str(i) for i in range(num_classes)]

    fn = f"{dir_res}/{base_name}_CM.png"
    plot_confusion_matrix(cm, fn, disp_labs, show)

    cm_norm = np.around(cm / (1e-10 + cm.astype(float).sum(axis=1)[:, np.newaxis]), decimals=2)
    fn_norm = f"{dir_res}/{base_name}_CM_norm.png"
    plot_confusion_matrix(cm_norm, fn_norm, disp_labs, show)

    precision, recall, fmean, iou = compute_f_mean_from_cm(cm)
    miou = mean_iou(cm)

    print(iou)
    print(miou)

    return precision, recall, fmean, iou, miou
