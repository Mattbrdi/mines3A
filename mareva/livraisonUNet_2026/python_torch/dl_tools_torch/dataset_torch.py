"""PyTorch Dataset / DataLoader wrapping the synthetic "dead leaves"
image + segmentation generator.

This replaces dl_tools.keras_image2image.DeadLeavesWithSegmGenerator
(a Keras-style Python generator yielding (X, Y) numpy batches).

The synthetic data is unlimited: instead of drawing one big fixed batch
(as the notebook's `generateDB` did with `next(datagen.flow(batch_size=N))`),
DeadLeavesSegDataset draws a *fresh* random sample every time it is asked
for an item. `length` only defines how many samples make up one epoch for
the DataLoader; the underlying process can produce as many distinct images
as you like.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dl_tools_torch.random_image_generator import DeadLeavesWithSegm


def to_one_hot(labels, num_classes):
    """Convert an integer label map into a one-hot, channel-first tensor/array.

    Arguments:
        labels: integer label map of shape (H, W) or, batched, (N, H, W).
            Accepts a numpy array or a torch tensor.
        num_classes: number of classes (channels) in the one-hot output.

    Returns:
        (num_classes, H, W) or (N, num_classes, H, W), same array type
        (numpy or torch) as the input, dtype float32. This is the layout
        used throughout dl_tools_torch (DeadLeavesSegDataset,
        FixedDeadLeavesSegDataset, model outputs, losses).
    """
    as_numpy = isinstance(labels, np.ndarray)
    t = torch.as_tensor(labels).long()
    oh = torch.nn.functional.one_hot(t, num_classes=num_classes).float()
    # one_hot appends the class dim last; move it to become the channel axis
    oh = oh.movedim(-1, -3) if t.dim() == 3 else oh.movedim(-1, 0)
    return oh.numpy() if as_numpy else oh


class DeadLeavesSegDataset(Dataset):
    """Map-style dataset generating dead-leaves images + segmentation masks
    on the fly.

    Arguments:
        x_size, y_size: image dimensions.
        rog_list: list of random object generator instances
            (ROG_disks, ROG_rings, ...).
        noise: noise generator instance (e.g. AdditiveGaussianNoise) or None.
        background_val: background grey level of the input image.
        norm: normalization constant applied to the input image (X / norm).
        length: number of samples considered to make up one "epoch". Since
            samples are drawn on demand, this is just a bookkeeping length,
            not a hard cap on how much distinct data exists.
        num_classes: if given, Y is returned one-hot encoded on this many
            classes, channel-first (C, H, W), ready for a softmax/CE head.
            If None (default), Y is returned as a single-channel float
            label map (1, H, W), matching the original notebook's binary /
            multi-label-as-single-channel setup.

    Returns from __getitem__:
        x: FloatTensor of shape (1, H, W), normalized image.
        y: FloatTensor of shape (1, H, W) or (num_classes, H, W).
    """

    def __init__(self, x_size, y_size, rog_list, noise=None, background_val=0,
                 norm=255, length=1000, num_classes=None):
        self.dl = DeadLeavesWithSegm(
            x_size, y_size, rog_list, noise=noise,
            background_val=background_val, shuffle=False, norm=norm,
        )
        self.norm = float(norm)
        self.length = length
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # idx is ignored: every call draws a brand new random sample.
        out = self.dl(number=1)
        im = out["images"][0].astype(np.float32) / self.norm
        seg = out["segm"][0].astype(np.int64)

        x = torch.from_numpy(im).unsqueeze(0)  # (1, H, W)

        if self.num_classes is not None:
            y = to_one_hot(seg, self.num_classes)  # (num_classes, H, W)
        else:
            y = torch.from_numpy(seg).unsqueeze(0).float()  # (1, H, W)

        return x, y


class FixedDeadLeavesSegDataset(Dataset):
    """Pre-generates a fixed set of samples ONCE (optionally with a seed)
    and always returns the same images afterwards.

    Use this for the validation/test sets (and, if you want strictly
    reproducible experiments like the original notebook, for the training
    set too): unlike DeadLeavesSegDataset, the data does not change between
    epochs or between calls to the DataLoader, so val_loss is comparable
    epoch to epoch, and runs using the same `seed` are comparable to each
    other.

    Arguments: same as DeadLeavesSegDataset, plus:
        n_samples: number of samples to pre-generate (fixed dataset size).
        seed: if given, numpy's global RNG is seeded before generation and
            restored to its previous state afterwards, so the same seed
            always reproduces the same images regardless of what else in
            the program has already consumed randomness.
    """

    def __init__(self, x_size, y_size, rog_list, noise=None, background_val=0,
                 norm=255, n_samples=1000, num_classes=None, seed=None):
        dl = DeadLeavesWithSegm(
            x_size, y_size, rog_list, noise=noise,
            background_val=background_val, shuffle=False, norm=norm,
        )

        if seed is not None:
            rng_state = np.random.get_state()
            np.random.seed(seed)
        try:
            out = dl(number=n_samples)
        finally:
            if seed is not None:
                np.random.set_state(rng_state)

        ims = np.stack(out["images"]).astype(np.float32) / float(norm)  # (N, H, W)
        segs = np.stack(out["segm"]).astype(np.int64)                    # (N, H, W)

        self.X = torch.from_numpy(ims).unsqueeze(1)  # (N, 1, H, W)

        if num_classes is not None:
            self.Y = to_one_hot(segs, num_classes)  # (N, num_classes, H, W)
        else:
            self.Y = torch.from_numpy(segs).unsqueeze(1).float()  # (N, 1, H, W)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def _worker_init_fn(worker_id):
    """Reseeds numpy in each DataLoader worker so that parallel workers
    (which fork the process and would otherwise share the same numpy RNG
    state) do not all draw identical random images."""
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)


def make_dataloader(x_size, y_size, rog_list, noise=None, background_val=0,
                     norm=255, length=1000, num_classes=None, batch_size=32,
                     shuffle=True, num_workers=0, **dataloader_kwargs):
    """Convenience wrapper: build a DeadLeavesSegDataset and wrap it in a
    ready-to-use torch.utils.data.DataLoader.

    Example:
        train_loader = make_dataloader(
            x_size=32, y_size=32, rog_list=l_rog, noise=noise,
            norm=255, length=myP.nb_train_samples,
            num_classes=len(myP.classList) + 1,  # + background
            batch_size=myP.batch_size, shuffle=True,
        )
        for x, y in train_loader:
            ...
    """
    ds = DeadLeavesSegDataset(
        x_size, y_size, rog_list, noise=noise, background_val=background_val,
        norm=norm, length=length, num_classes=num_classes,
    )
    kwargs = dict(dataloader_kwargs)
    if num_workers > 0:
        kwargs.setdefault("worker_init_fn", _worker_init_fn)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, **kwargs)


def make_fixed_dataloader(x_size, y_size, rog_list, noise=None, background_val=0,
                           norm=255, n_samples=1000, num_classes=None, batch_size=32,
                           shuffle=True, seed=None, **dataloader_kwargs):
    """Like make_dataloader, but backed by FixedDeadLeavesSegDataset: the
    same `n_samples` images are generated once (deterministically if `seed`
    is given) and reused identically at every epoch / every call.

    Use this for validation and test sets so that val_loss is comparable
    across epochs, and, with a fixed `seed`, across separate runs/executions
    (e.g. when comparing hyperparameters)."""
    ds = FixedDeadLeavesSegDataset(
        x_size, y_size, rog_list, noise=noise, background_val=background_val,
        norm=norm, n_samples=n_samples, num_classes=num_classes, seed=seed,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **dataloader_kwargs)
