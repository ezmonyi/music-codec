"""
Multi-Scale STFT Discriminator (EnCodec-style).
Operates on waveform (B, 1, T). Use after mel_to_waveform for mel-based codec.
"""

import typing as tp
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torchaudio

# Suppress FutureWarning for torch.nn.utils.weight_norm deprecation
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

try:
    from einops import rearrange
except ImportError:
    def rearrange(x, pattern, **axes):
        if pattern == "b c w t -> b c t w":
            return x.permute(0, 1, 3, 2)
        if pattern == "(b t) -> b t":
            b = axes.get("b")
            return x.reshape(b, -1)
        raise NotImplementedError("einops not installed; pip install einops")


def get_2d_padding(kernel_size: tp.Tuple[int, int], dilation: tp.Tuple[int, int] = (1, 1)):
    return (
        ((kernel_size[0] - 1) * dilation[0]) // 2,
        ((kernel_size[1] - 1) * dilation[1]) // 2,
    )


def NormConv2d(*args, norm: str = "weight_norm", **kwargs) -> nn.Module:
    conv = nn.Conv2d(*args, **kwargs)
    if norm == "weight_norm":
        conv = weight_norm(conv)
    return conv


class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator: spectrogram -> conv stack -> logits."""

    def __init__(
        self,
        filters: int,
        in_channels: int = 1,
        out_channels: int = 1,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        max_filters: int = 1024,
        filters_scale: int = 1,
        kernel_size: tp.Tuple[int, int] = (3, 9),
        dilations: tp.List[int] = (1, 2, 4),
        stride: tp.Tuple[int, int] = (1, 2),
        normalized: bool = True,
        norm: str = "weight_norm",
        activation: str = "LeakyReLU",
        activation_params: dict = None,
    ):
        super().__init__()
        if activation_params is None:
            activation_params = {"negative_slope": 0.2}
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=self.normalized,
            center=False,
            pad_mode=None,
            power=None,
        )
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(
                spec_channels, self.filters,
                kernel_size=kernel_size,
                padding=get_2d_padding(kernel_size),
                norm=norm,
            )
        )
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale ** (i + 1)) * self.filters, max_filters)
            self.convs.append(
                NormConv2d(
                    in_chs, out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(kernel_size, (dilation, 1)),
                    norm=norm,
                )
            )
            in_chs = out_chs
        out_chs = min((filters_scale ** (len(dilations) + 1)) * self.filters, max_filters)
        self.convs.append(
            NormConv2d(
                in_chs, out_chs,
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                norm=norm,
            )
        )
        self.conv_post = NormConv2d(
            out_chs, self.out_channels,
            kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])),
            norm=norm,
        )

    def forward(self, x: torch.Tensor):
        """x: (B, 1, T) or (B, T). Returns logits, list of feature maps."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        fmap = []
        z = self.spec_transform(x)
        z = torch.cat([z.real, z.imag], dim=1)
        z = rearrange(z, "b c w t -> b c t w")
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            fmap.append(z)
        z = self.conv_post(z)
        return z, fmap


class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-scale STFT discriminators."""

    def __init__(
        self,
        filters: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        n_ffts: tp.List[int] = (1024, 2048, 512),
        hop_lengths: tp.List[int] = (256, 512, 128),
        win_lengths: tp.List[int] = (1024, 2048, 512),
        **kwargs,
    ):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(
                filters,
                in_channels=in_channels,
                out_channels=out_channels,
                n_fft=n_ffts[i],
                hop_length=hop_lengths[i],
                win_length=win_lengths[i],
                **kwargs,
            )
            for i in range(len(n_ffts))
        ])
        self.num_discriminators = len(self.discriminators)

    def forward(self, x: torch.Tensor) -> tp.Tuple[tp.List[torch.Tensor], tp.List[tp.List[torch.Tensor]]]:
        """x: (B, 1, T). Returns list of logits per scale, list of fmaps per scale."""
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps
