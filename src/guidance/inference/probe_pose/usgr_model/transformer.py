"""Transformer components with 2D Axial RoPE.

Vendored verbatim from USGR ``usgr/layers/transformer.py`` (no functional changes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
import einops


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weight, 1)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU FFN
    """

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        out_features = out_features or in_features
        # 8/3 ratio to match param count of standard MLP with ratio=4
        hidden_features = int(in_features * 8 / 3)
        # Round to multiple of 256 for efficiency
        hidden_features = ((hidden_features + 255) // 256) * 256

        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)  # Gate
        self.w2 = nn.Linear(hidden_features, out_features, bias=bias)  # Down
        self.w3 = nn.Linear(in_features, hidden_features, bias=bias)  # Up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RoPE(nn.Module):
    """
    2D Rotary Position Embedding
    """

    def __init__(
        self,
        head_dim: int,
        theta: float = 100.0,
    ):
        """
        Args:
            head_dim: Dimension per attention head (embed_dim // num_heads)
            theta: Base frequency (DINOv3 uses 100.0)
        """
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

        # Half dimensions for each axis (axial = separate x and y)
        self.dim_y = head_dim // 2
        self.dim_x = head_dim - self.dim_y

        # Precompute inverse frequencies for each axis
        # freq_i = theta^(-2i/d) for i in [0, d/2)
        inv_freq_y = 1.0 / (
            theta ** (torch.arange(0, self.dim_y, 2).float() / self.dim_y)
        )
        inv_freq_x = 1.0 / (
            theta ** (torch.arange(0, self.dim_x, 2).float() / self.dim_x)
        )

        self.register_buffer("inv_freq_y", inv_freq_y, persistent=False)
        self.register_buffer("inv_freq_x", inv_freq_x, persistent=False)

    def _compute_rope(
        self,
        positions: torch.Tensor,  # [N] normalized positions in [-1, 1]
        inv_freq: torch.Tensor,  # [d/4] inverse frequencies
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin for one axis."""
        # [N, d/4] = [N, 1] * [1, d/4]
        angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)
        # Duplicate for pairs: [N, d/2]
        angles = angles.repeat_interleave(2, dim=-1)
        return angles.cos(), angles.sin()

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate pairs: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]"""
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        h: int,
        w: int,
        num_prefix_tokens: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply 2D axial RoPE (up to prefix tokens).

        Args:
            q, k: [B, heads, N, head_dim] where N = num_prefix_tokens + h * w
            h, w: Spatial dimensions in patches
            num_prefix_tokens: Number of prefix tokens (e.g., CLS/pose) to skip

        Returns:
            Rotated q, k (prefix tokens unchanged, spatial tokens rotated)
        """
        device, dtype = q.device, q.dtype

        # split off prefix tokens
        if num_prefix_tokens > 0:
            q_prefix = q[:, :, :num_prefix_tokens, :]
            k_prefix = k[:, :, :num_prefix_tokens, :]
            q_spatial = q[:, :, num_prefix_tokens:, :]
            k_spatial = k[:, :, num_prefix_tokens:, :]
        else:
            q_spatial = q
            k_spatial = k

        # Create normalized coordinates in [-1, 1]
        y_coords = torch.linspace(-1, 1, h, device=device, dtype=dtype)
        x_coords = torch.linspace(-1, 1, w, device=device, dtype=dtype)

        # Create grid and flatten: [h, w] -> [h*w]
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        pos_y = grid_y.flatten()  # [N_spatial]
        pos_x = grid_x.flatten()  # [N_spatial]

        # Compute cos/sin for each axis
        cos_y, sin_y = self._compute_rope(pos_y, self.inv_freq_y)  # [N_spatial, dim_y]
        cos_x, sin_x = self._compute_rope(pos_x, self.inv_freq_x)  # [N_spatial, dim_x]

        # Concatenate: first half for y, second half for x (axial)
        cos = torch.cat([cos_y, cos_x], dim=-1)  # [N_spatial, head_dim]
        sin = torch.cat([sin_y, sin_x], dim=-1)  # [N_spatial, head_dim]

        # Add batch and head dimensions: [1, 1, N_spatial, head_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotation
        q_spatial_rot = q_spatial * cos + self._rotate_half(q_spatial) * sin
        k_spatial_rot = k_spatial * cos + self._rotate_half(k_spatial) * sin

        # Recombine spatial and prefix tokens
        if num_prefix_tokens > 0:
            q_rot = torch.cat([q_prefix, q_spatial_rot], dim=2)
            k_rot = torch.cat([k_prefix, k_spatial_rot], dim=2)
        else:
            q_rot = q_spatial_rot
            k_rot = k_spatial_rot

        return q_rot, k_rot


class Attention(nn.Module):
    """Multi-head self-attention with RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope_theta: float = 100.0,
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = RoPE(
            head_dim=self.head_dim,
            theta=rope_theta,
        )

    def forward(
        self, x: torch.Tensor, h: int, w: int, num_prefix_tokens: int = 0
    ) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x)  # [B, N, 3 * C]
        q, k, v = einops.rearrange(
            qkv,
            "b n (qkv h d) -> qkv b h n d",
            qkv=3,
            h=self.num_heads,
            d=self.head_dim,
        ).unbind(0)

        # Apply RoPE
        q, k = self.rope(q, k, h, w, num_prefix_tokens=num_prefix_tokens)

        # SDPA
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
            scale=self.scale,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TransformerBlock(nn.Module):
    """Transformer block with RoPE attention and SwiGLU FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: type = RMSNorm,
        init_values: Optional[float] = None,
        rope_theta: float = 100.0,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            rope_theta=rope_theta,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = SwiGLU(in_features=dim, out_features=dim, bias=ffn_bias)

    def forward(
        self, x: torch.Tensor, h: int, w: int, num_prefix_tokens: int = 0
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), h, w, num_prefix_tokens=num_prefix_tokens)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerHead(nn.Module):
    """Lightweight transformer head with 2D Axial RoPE for DINOv3 features."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        out_channels: Optional[int] = None,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: type = RMSNorm,
        init_values: Optional[float] = 1e-5,
        use_pose_token: bool = True,
        use_conditioning_token: bool = False,
        conditioning_dim: int = 12,
        output_spatial: bool = True,
        rope_theta: float = 100.0,
        pose_out_channels: int = 12,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.out_channels = out_channels or embed_dim
        self.use_pose_token = use_pose_token
        self.output_spatial = output_spatial
        self.num_prefix_tokens = use_pose_token + use_conditioning_token
        self.use_conditioning_token = use_conditioning_token
        self.conditioning_dim = conditioning_dim
        self.use_pose_token = use_pose_token

        # Input projection
        self.input_proj = nn.Linear(in_channels, embed_dim)

        # Optional POSE token
        if self.use_pose_token:
            self.pose_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.pose_token, std=0.02)
            self.pose_proj = nn.Linear(embed_dim, pose_out_channels)

        # Optional Conditioning token (projects conditioning_dim to embed_dim)
        if self.use_conditioning_token:
            self.conditioning_proj = nn.Linear(conditioning_dim, embed_dim)

        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    rope_theta=rope_theta,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)
        self.output_proj = (
            nn.Linear(embed_dim, self.out_channels)
            if self.out_channels != embed_dim
            else nn.Identity()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        return_pose: bool = True,
    ) -> "torch.Tensor | tuple[torch.Tensor, torch.Tensor]":
        B, C, H, W = x.shape

        x = einops.rearrange(x, "b c h w -> b (h w) c")
        x = self.input_proj(x)

        # Conditioning token handling
        if self.use_conditioning_token:
            if y is None:
                y_proj = torch.zeros(B, 1, self.embed_dim, device=x.device, dtype=x.dtype)
            else:
                assert len(y.shape) == 2, f'conditioning vector must be a 2D tensor with shape [B, {self.conditioning_dim}], got {y.shape}'
                assert y.shape[1] == self.conditioning_dim, f'conditioning vector must have dim {self.conditioning_dim}, got {y.shape[1]}'
                y_proj = self.conditioning_proj(y).unsqueeze(1)  # [B, S=1, embed_dim]
            x = torch.cat([y_proj, x], dim=1)

        # POSE token handling
        if self.use_pose_token:
            pose_tokens = self.pose_token.expand(B, -1, -1)
            x = torch.cat([pose_tokens, x], dim=1)

        # Apply blocks
        for block in self.blocks:
            x = block(x, H, W, num_prefix_tokens=self.num_prefix_tokens)

        x = self.norm(x)

        if self.use_pose_token:
            pose_token = x[:, 0]
            x = x[:, 1:]

        if self.use_conditioning_token:
            x = x[:, 1:]  # remove conditioning token

        x = self.output_proj(x)

        if self.output_spatial:
            x = einops.rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        if return_pose and self.use_pose_token:
            pose_out = self.pose_proj(pose_token)
            return x, pose_out

        return x
