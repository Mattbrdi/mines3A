"""Modular U-Net, PyTorch port of dl_tools.modular_unet.

Faithfully mirrors the channel-count bookkeeping of the original Keras
`modular_u_net` / `att_unet_mod_lev` functions: encoder doubles the filter
count at every downsampling step, decoder halves it at every upsampling
step, with optional residual blocks, batch norm, conv- or pooling-based
sampling, skip connections and (for the attention variant) additive
attention gates on the skip connections.

Note: as in the original, input spatial dimensions should be a multiple of
2 ** (nb_levels - 1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetConvBlock(nn.Module):
    """Two 3x3 conv layers, optionally batch-normalized, with an optional
    residual (1x1-style 3x3) skip added before the final activation --
    port of dl_tools.modular_unet.unet_block."""

    def __init__(self, in_channels, out_channels, res=False, batch_norm=False):
        super().__init__()
        self.res = res
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        if res:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.res:
            out = out + self.res_conv(x)
        return F.relu(out)


class DownSample(nn.Module):
    """Strided conv or max-pooling downsampling -- port of unet_down."""

    def __init__(self, conv_sampling, in_channels=None, out_channels=None):
        super().__init__()
        if conv_sampling:
            self.op = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                 stride=2, padding=1)
        else:
            self.op = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        return self.op(x)


class UpSample(nn.Module):
    """Transposed conv or nearest-neighbour upsampling -- port of unet_up."""

    def __init__(self, conv_sampling, in_channels=None, out_channels=None):
        super().__init__()
        if conv_sampling:
            self.op = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,
                                          stride=2, padding=1, output_padding=1)
        else:
            self.op = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        return self.op(x)


class ModularUNet(nn.Module):
    """Note that the dimensions of the input images should be
    multiples of 16.


    nb_levels: U "depth", number of downsampling and upsampling steps
    nb_filters_0 : initial number of filters in the convolutional layer.
    output_channels: number of output channels.
    sigma_noise: standard deviation of the gaussian noise layer. If equal to zero, this layer is deactivated.
    drop: dropout rate.
    skip: boolean indicating if skip connections should be used.
    res: boolean indicating if residual blocks should be used.
    conv_sampling: boolean indicating if sub- and up-sampling should be
        done with convolutions
    batch_norm: should we use batch normalization?

    Returns:
    U-Net model - it still needs to be compiled.

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Olaf Ronneberger, Philipp Fischer, Thomas Brox
    MICCAI 2015

    output_activation: "sigmoid" (default) for a single "object present"
        channel (binary segmentation), or "softmax" for
        `output_channels` mutually exclusive classes (background
        included) -- use this when there is more than one foreground
        class, paired with one-hot targets and a CE/soft-Dice loss over
        the channel dimension.

    """

    def __init__(self, in_channels=1, nb_levels=4, nb_filters_0=32,
                 output_channels=1, sigma_noise=0.0, drop=0.0, skip=True,
                 res=False, conv_sampling=False, batch_norm=False,
                 output_activation="sigmoid"):
        super().__init__()
        self.nb_levels = nb_levels
        self.skip = skip
        self.sigma_noise = sigma_noise
        if output_activation not in ("sigmoid", "softmax"):
            raise ValueError("output_activation must be 'sigmoid' or 'softmax'")
        self.output_activation = output_activation

        self.down_blocks = nn.ModuleList()
        self.down_samplers = nn.ModuleList()
        self.enc_channels = []

        nb_filters = nb_filters_0
        in_ch = in_channels
        for lev in range(nb_levels):
            self.down_blocks.append(
                UNetConvBlock(in_ch, nb_filters, res=res, batch_norm=batch_norm))
            self.enc_channels.append(nb_filters)
            if lev < nb_levels - 1:
                self.down_samplers.append(DownSample(conv_sampling, nb_filters, nb_filters))
                in_ch = nb_filters
                nb_filters *= 2

        self.dropout = nn.Dropout2d(drop) if drop > 0.0 else nn.Identity()

        self.up_samplers = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for lev in range(nb_levels - 1):
            self.up_samplers.append(UpSample(conv_sampling, nb_filters, nb_filters))
            skip_ch = self.enc_channels[nb_levels - 2 - lev] if skip else 0
            in_ch = nb_filters + skip_ch
            nb_filters = nb_filters // 2
            self.up_blocks.append(
                UNetConvBlock(in_ch, nb_filters, res=res, batch_norm=batch_norm))

        self.out_conv = nn.Conv2d(nb_filters, output_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        out = x
        for lev in range(self.nb_levels):
            out = self.down_blocks[lev](out)
            if lev < self.nb_levels - 1:
                skips.append(out)
                out = self.down_samplers[lev](out)

        out = self.dropout(out)

        for lev in range(self.nb_levels - 1):
            out = self.up_samplers[lev](out)
            if self.skip:
                out = torch.cat([out, skips.pop()], dim=1)
            out = self.up_blocks[lev](out)

        if self.sigma_noise > 0 and self.training:
            out = out + torch.randn_like(out) * self.sigma_noise

        logits = self.out_conv(out)
        if self.output_activation == "softmax":
            return torch.softmax(logits, dim=1)
        return torch.sigmoid(logits)


class AttentionGate2D(nn.Module):
    """Additive attention gate -- port of attention_block_2d.

    Oktay et al., "Attention U-Net: Learning Where to Look for the
    Pancreas", 2018.
    """

    def __init__(self, x_channels, g_channels, inter_channels):
        super().__init__()
        self.theta_x = nn.Conv2d(x_channels, inter_channels, kernel_size=1)
        self.phi_g = nn.Conv2d(g_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)

    def forward(self, x, g):
        f = F.relu(self.theta_x(x) + self.phi_g(g))
        rate = torch.sigmoid(self.psi(f))
        return x * rate


class AttentionModularUNet(nn.Module):
    """Port of dl_tools.modular_unet.att_unet_mod_lev.

    When `attention=True`, `skip` should also be True (the attention gate
    operates on the skip connection, matching the original Keras logic).
    """

    def __init__(self, in_channels=1, nb_levels=4, nb_filters_0=32,
                 output_channels=1, sigma_noise=0.0, drop=0.0, skip=True,
                 res=False, conv_sampling=False, batch_norm=False, attention=False,
                 output_activation="sigmoid"):
        super().__init__()
        self.nb_levels = nb_levels
        self.skip = skip
        self.attention = attention
        self.sigma_noise = sigma_noise
        if output_activation not in ("sigmoid", "softmax"):
            raise ValueError("output_activation must be 'sigmoid' or 'softmax'")
        self.output_activation = output_activation

        self.down_blocks = nn.ModuleList()
        self.down_samplers = nn.ModuleList()
        self.enc_channels = []

        nb_filters = nb_filters_0
        in_ch = in_channels
        for lev in range(nb_levels):
            self.down_blocks.append(
                UNetConvBlock(in_ch, nb_filters, res=res, batch_norm=batch_norm))
            self.enc_channels.append(nb_filters)
            if lev < nb_levels - 1:
                self.down_samplers.append(DownSample(conv_sampling, nb_filters, nb_filters))
                in_ch = nb_filters
                nb_filters *= 2

        self.dropout = nn.Dropout2d(drop) if drop > 0.0 else nn.Identity()

        self.up_samplers = nn.ModuleList()
        self.att_gates = nn.ModuleList() if attention else None
        self.up_blocks = nn.ModuleList()

        for lev in range(nb_levels - 1):
            self.up_samplers.append(UpSample(conv_sampling, nb_filters, nb_filters))
            skip_ch = self.enc_channels[nb_levels - 2 - lev]
            if attention:
                self.att_gates.append(AttentionGate2D(skip_ch, nb_filters, max(nb_filters // 4, 1)))
            in_ch = nb_filters + skip_ch if skip else nb_filters
            nb_filters = nb_filters // 2
            self.up_blocks.append(
                UNetConvBlock(in_ch, nb_filters, res=res, batch_norm=batch_norm))

        self.out_conv = nn.Conv2d(nb_filters, output_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        out = x
        for lev in range(self.nb_levels):
            out = self.down_blocks[lev](out)
            if lev < self.nb_levels - 1:
                skips.append(out)
                out = self.down_samplers[lev](out)

        out = self.dropout(out)

        for lev in range(self.nb_levels - 1):
            g = self.up_samplers[lev](out)
            skip = skips.pop()
            if self.attention:
                att_x = self.att_gates[lev](skip, g)
                out = torch.cat([g, att_x], dim=1) if self.skip else g
            elif self.skip:
                out = torch.cat([g, skip], dim=1)
            else:
                out = g
            out = self.up_blocks[lev](out)

        if self.sigma_noise > 0 and self.training:
            out = out + torch.randn_like(out) * self.sigma_noise

        logits = self.out_conv(out)
        if self.output_activation == "softmax":
            return torch.softmax(logits, dim=1)
        return torch.sigmoid(logits)
