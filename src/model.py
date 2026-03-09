# src/model.py
"""Model creation utilities: pretrained HF DDPM or custom UNet."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel

from src.config import config, device, USE_PRETRAINED


# ===================== Custom UNet Components =====================

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device) * np.log(10000) / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, in_c)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.time_mlp = nn.Linear(time_dim, out_c)
        if in_c != out_c:
            self.skip = nn.Conv2d(in_c, out_c, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + self.time_mlp(t)[..., None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3,
                 base_channels=64, channel_mults=None,
                 num_res_blocks=2, time_dim=128):
        super().__init__()
        if channel_mults is None:
            channel_mults = [1, 2, 2, 4]
        self.time_embed = TimeEmbedding(time_dim)
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.downs = nn.ModuleList()
        now_c = base_channels
        for i, mult in enumerate(channel_mults):
            out_c = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(now_c, out_c, time_dim))
                now_c = out_c
            if i != len(channel_mults) - 1:
                self.downs.append(
                    nn.Conv2d(now_c, now_c, 3, stride=2, padding=1)
                )
        self.mid = nn.ModuleList([
            ResBlock(now_c, now_c, time_dim),
            ResBlock(now_c, now_c, time_dim)
        ])
        self.ups = nn.ModuleList()
        for i in reversed(range(len(channel_mults))):
            mult = channel_mults[i]
            skip_c = base_channels * mult
            if i < len(channel_mults) - 1:
                self.ups.append(nn.Upsample(scale_factor=2, mode='nearest'))
            for _ in range(num_res_blocks):
                self.ups.append(ResBlock(now_c + skip_c, skip_c, time_dim))
                now_c = skip_c
        self.out_norm = nn.GroupNorm(8, now_c)
        self.out_conv = nn.Conv2d(now_c, out_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        h = self.in_conv(x)
        skips = []
        for layer in self.downs:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
                skips.append(h)
            else:
                h = layer(h)
        for layer in self.mid:
            h = layer(h, t_emb)
        for layer in self.ups:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                h = layer(h, t_emb)
            else:
                h = layer(h)
        h = self.out_norm(h)
        h = F.silu(h)
        return self.out_conv(h)


# ===================== Factory =====================

def create_model():
    """Create and return the diffusion model on the configured device."""
    if USE_PRETRAINED:
        model = UNet2DModel.from_pretrained("google/ddpm-cifar10-32")
        model = model.to(device)
        return model
    else:
        return UNet(
            in_channels=config['channels'],
            out_channels=config['channels'],
            base_channels=config['base_channels'],
            channel_mults=config['channel_mults'],
            num_res_blocks=config['num_res_blocks']
        ).to(device)
