"""Dependency-light inference fallbacks for Pluto's optional vision layers."""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, value):
        if self.drop_prob == 0.0 or not self.training:
            return value
        keep_prob = 1.0 - self.drop_prob
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        mask = value.new_empty(shape).bernoulli_(keep_prob)
        return value * mask / keep_prob


class NeighborhoodAttention1D(nn.Module):
    """Pure PyTorch NATTEN-compatible 1-D attention for inference.

    Parameter names and shapes match ``natten.NeighborhoodAttention1D`` so
    official Pluto checkpoints load without conversion. The implementation is
    intentionally simple and is used only when the compiled NATTEN package is
    unavailable.
    """

    def __init__(
        self,
        dim,
        kernel_size=7,
        dilation=None,
        num_heads=1,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = int(dim)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation or 1)
        if self.dilation != 1:
            raise NotImplementedError("Pluto fallback is validated only for dilation=1")
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.scale = float(qk_scale or self.head_dim ** -0.5)
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(
            torch.zeros(self.num_heads, 2 * self.kernel_size - 1)
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, value):
        batch, length, _ = value.shape
        if length < self.kernel_size:
            raise ValueError("Sequence shorter than kernel; padded NATTEN path is not implemented")
        qkv = self.qkv(value).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, val = qkv.unbind(dim=2)
        outputs = []
        radius = self.kernel_size // 2
        max_start = max(length - self.kernel_size * self.dilation, 0)
        for index in range(length):
            start = min(max(index - radius * self.dilation, 0), max_start)
            neighbors = torch.arange(
                start,
                min(start + self.kernel_size * self.dilation, length),
                self.dilation,
                device=value.device,
            )
            selected_key = key[:, neighbors].transpose(1, 2)
            selected_val = val[:, neighbors].transpose(1, 2)
            scores = torch.einsum(
                "bhd,bhkd->bhk", query[:, index] * self.scale, selected_key
            )
            relative = (neighbors - index) // self.dilation + self.kernel_size - 1
            relative = relative.clamp(0, 2 * self.kernel_size - 2)
            scores = scores + self.rpb[:, relative].unsqueeze(0)
            attention = self.attn_drop(scores.softmax(dim=-1))
            outputs.append(
                torch.einsum("bhk,bhkd->bhd", attention, selected_val)
            )
        output = torch.stack(outputs, dim=1).reshape(batch, length, self.dim)
        return self.proj_drop(self.proj(output))
