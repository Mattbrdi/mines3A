"""Training / visualization utilities, PyTorch port of dl_tools.tools.

Replaces model.fit / Keras callbacks (ModelCheckpoint, EarlyStopping,
CSVLogger) with an explicit training loop offering the same behaviour:
best-checkpoint saving on val_loss and early stopping on patience.
"""
import os
import csv
from shutil import copyfile

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from dl_tools_torch.losses_torch import jaccard2_loss, binary_focal_loss

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def set_optimizer(model, myP):
    """Port of dl_tools.tools.setOptimizer."""
    if myP.opt_name == "sgd":
        myP.opt = optim.SGD(model.parameters(), lr=myP.lr)
    elif myP.opt_name == "rmsprop":
        myP.opt = optim.RMSprop(model.parameters(), lr=myP.lr)
    elif myP.opt_name == "adagrad":
        myP.opt = optim.Adagrad(model.parameters(), lr=myP.lr)
    elif myP.opt_name == "adadelta":
        myP.opt = optim.Adadelta(model.parameters(), lr=myP.lr)
    elif myP.opt_name == "adam":
        myP.opt = optim.Adam(model.parameters(), lr=myP.lr)
    else:
        raise NameError("Wrong optimizer name")
    return myP.opt


def set_loss(myP):
    """Port of dl_tools.tools.setLoss."""
    if myP.lossF == "jac" or myP.lossF == "jac2":
        myP.loss_func = jaccard2_loss
        myP.loss_str = "_j2"
    elif myP.lossF == "cce":
        def cce(y_true, y_pred):
            eps = 1e-7
            y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
            return -(y_true * torch.log(y_pred)).sum(dim=1).mean()
        myP.loss_func = cce
        myP.loss_str = "_cce"
    elif myP.lossF == "focal":
        myP.loss_func = binary_focal_loss()
        myP.loss_str = "_focal"
    elif myP.lossF == "mse":
        myP.loss_func = torch.nn.functional.mse_loss
        myP.loss_str = "_mse"
    else:
        raise ValueError("Unknown loss function: %s" % myP.lossF)
    return myP.loss_func


def train_model(model, train_loader, val_loader, myP, device="cpu", verbose=True):
    """Trains `model`, mirroring model.fit(..., callbacks=[ModelCheckpoint,
    EarlyStopping, CSVLogger]) from the original notebook.

    Returns a history dict: {"loss": [...], "val_loss": [...]}, with the
    same keys as Keras' history.history used later for plotting.
    """
    os.makedirs(myP.dirRes, exist_ok=True)
    model.to(device)
    optimizer = set_optimizer(model, myP)
    loss_fn = set_loss(myP)

    history = {"loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    model_file_path = os.path.join(myP.dirRes, myP.baseName + "_best.pt")
    csv_path = os.path.join(myP.dirRes, "epochs.csv")
    with open(csv_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "loss", "val_loss"])

        for epoch in range(myP.nb_epoch):
            model.train()
            running_loss, n_batches = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = loss_fn(y, y_pred)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1
            train_loss = running_loss / max(n_batches, 1)

            model.eval()
            val_loss_sum, n_val = 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    y_pred = model(x)
                    val_loss_sum += loss_fn(y, y_pred).item()
                    n_val += 1
            val_loss = val_loss_sum / max(n_val, 1)

            history["loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            csv_writer.writerow([epoch, train_loss, val_loss])
            csv_file.flush()

            if verbose:
                print("Epoch %d/%d - loss: %.5f - val_loss: %.5f" %
                      (epoch + 1, myP.nb_epoch, train_loss, val_loss))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, model_file_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= myP.patience:
                    if verbose:
                        print("Early stopping at epoch %d" % (epoch + 1))
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def predict(model, loader, device="cpu"):
    """Runs `model` over every batch of `loader` and concatenates outputs,
    mirroring model.predict(X_val)."""
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            preds.append(model(x).cpu().numpy())
    return np.concatenate(preds, axis=0)


def collect_dataset(loader):
    """Concatenates every (x, y) batch of `loader` into two numpy arrays.

    Use this instead of `next(iter(loader))` whenever you need X/Y aligned
    with `predict(model, loader, ...)`'s output: `next(iter(loader))` only
    returns the FIRST batch, while `predict` iterates over the whole loader
    and concatenates -- mixing the two gives mismatched array sizes.
    """
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.numpy())
        ys.append(y.numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def plot_learning_curves(history, out_path, show=False):
    plt.plot(range(len(history["loss"])), history["loss"], label="train")
    plt.plot(range(len(history["val_loss"])), history["val_loss"], label="val")
    plt.title("Training performance")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()
    if show:
        plt.show()
    else:
        plt.savefig(out_path)
    plt.clf()


def backup_files(this_file_name, myP, extra_files=("parameters_torch.py",)):
    """Port of dl_tools.tools.backupFiles."""
    os.makedirs(myP.dirRes, exist_ok=True)
    if os.path.exists(this_file_name):
        copyfile(this_file_name, os.path.join(myP.dirRes, os.path.basename(this_file_name)))
    for f in extra_files:
        if os.path.exists(f):
            copyfile(f, os.path.join(myP.dirRes, os.path.basename(f)))


def visualize_batch(nb_plots, X, Y, show=True, out_prefix=None):
    """Port of dl_tools.tools.visualizeTraining. X, Y are channel-first
    (N, C, H, W) numpy arrays, as produced by DeadLeavesSegDataset."""
    for index in range(min(nb_plots, len(X))):
        plt.subplot(1, 2, 1)
        plt.imshow(X[index, 0])
        plt.title("Image")
        plt.subplot(1, 2, 2)
        seg = Y[index, 0] if Y.shape[1] == 1 else np.argmax(Y[index], axis=0)
        plt.imshow(seg)
        plt.title("Segm")
        if show:
            plt.show()
        elif out_prefix:
            plt.savefig(f"{out_prefix}_{index:02d}.png")
        plt.clf()


def visualize_val(nb_plots, X_val, Y_val, Y_pred, dir_res, base_name, show=False):
    """Port of dl_tools.tools.visualizeVal. All arrays channel-first
    (N, C, H, W). Works for both binary (C=1) and multi-class (C>1, one-hot
    / softmax) Y_val and Y_pred -- multi-channel arrays are collapsed to a
    label map via argmax before display."""
    Y_gt = np.argmax(Y_val, axis=1) if Y_val.shape[1] > 1 else Y_val[:, 0]
    Y_pred_disp = np.argmax(Y_pred, axis=1) if Y_pred.shape[1] > 1 else Y_pred[:, 0]

    for index in range(min(nb_plots, len(X_val))):
        imori = X_val[index, 0] * 255
        imgt = Y_gt[index] * 100
        impred = Y_pred_disp[index] * 100
        plt.subplot(1, 3, 1)
        plt.imshow(imori, vmin=0, vmax=255)
        plt.title("original")
        plt.subplot(1, 3, 2)
        plt.imshow(imgt, vmin=0, vmax=255)
        plt.title("GT")
        plt.subplot(1, 3, 3)
        plt.imshow(impred, vmin=0, vmax=255)
        plt.title("Prediction")
        if show:
            plt.show()
        else:
            plt.savefig(os.path.join(dir_res, f"{base_name}_{index:02d}.png"))
        plt.clf()
import time
def get_time_string():
    """ return time in string format"""
    time_list = time.localtime()
    return "%s_" % str(time_list[0]) + "_".join(["%02d" % i for i in time_list[1:6]])
