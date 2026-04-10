"""
Adapter modules for efficient fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Adapter(nn.Module):
    """
    Bottleneck adapter for efficient fine-tuning.

    Architecture: input -> down_proj -> activation -> up_proj -> output
    The output is added as a residual to the original features.
    """

    def __init__(
        self,
        in_dim: int,
        bottleneck_dim: int = 64,
        activation: str = "gelu",
        init_scale: float = 1e-3,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.bottleneck_dim = bottleneck_dim

        # Down projection
        self.down_proj = nn.Linear(in_dim, bottleneck_dim)

        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

        # Up projection
        self.up_proj = nn.Linear(bottleneck_dim, in_dim)

        # Initialize with small values for stable training
        self._init_weights(init_scale)

    def _init_weights(self, scale: float):
        """Initialize weights with small values"""
        nn.init.normal_(self.down_proj.weight, std=scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.normal_(self.up_proj.weight, std=scale)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (..., in_dim)

        Returns:
            Adapter output of same shape as input
        """
        down = self.down_proj(x)
        act = self.activation(down)
        up = self.up_proj(act)
        return up


class AdapterCLIP(nn.Module):
    """
    CLIP model with adapters inserted into transformer blocks.
    This is an alternative implementation that directly modifies the forward pass.
    """

    def __init__(
        self,
        clip_model: nn.Module,
        adapter_dim: int = 64,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.clip_model = clip_model
        self.adapter_dim = adapter_dim

        # Freeze backbone
        if freeze_backbone:
            for param in clip_model.parameters():
                param.requires_grad = False

        # Get dimensions
        self.visual_dim = clip_model.visual.transformer.width
        self.text_dim = clip_model.transformer.width

        # Create adapters for visual encoder
        num_visual_layers = len(clip_model.visual.transformer.resblocks)
        self.visual_adapters = nn.ModuleList([
            Adapter(self.visual_dim, adapter_dim)
            for _ in range(num_visual_layers)
        ])

        # Create adapters for text encoder
        num_text_layers = len(clip_model.transformer.resblocks)
        self.text_adapters = nn.ModuleList([
            Adapter(self.text_dim, adapter_dim)
            for _ in range(num_text_layers)
        ])

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image with adapters"""
        # Patch embedding
        x = self.clip_model.visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # Add class token and position embedding
        x = torch.cat([
            self.clip_model.visual.class_embedding.to(x.dtype) +
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)

        x = self.clip_model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # Transformer blocks with adapters
        for i, block in enumerate(self.clip_model.visual.transformer.resblocks):
            x = block(x)
            # Apply adapter
            x = x + self.visual_adapters[i](x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.visual.ln_post(x[:, 0, :])

        if self.clip_model.visual.proj is not None:
            x = x @ self.clip_model.visual.proj

        return x

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text with adapters"""
        x = self.clip_model.token_embedding(text)
        x = x + self.clip_model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND

        # Transformer blocks with adapters
        for i, block in enumerate(self.clip_model.transformer.resblocks):
            x = block(x)
            # Apply adapter
            x = x + self.text_adapters[i](x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x)

        # Take features from EOT token
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_model.text_projection

        return x

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ):
        """Forward pass"""
        output = {}

        if image is not None:
            image_features = self.encode_image(image)
            image_features = F.normalize(image_features, dim=-1)
            output['image_features'] = image_features

        if text is not None:
            text_features = self.encode_text(text)
            text_features = F.normalize(text_features, dim=-1)
            output['text_features'] = text_features

        return output

    def get_adapter_params(self):
        """Get adapter parameters only"""
        params = []
        params.extend(self.visual_adapters.parameters())
        params.extend(self.text_adapters.parameters())
        return params

    def get_adapter_state_dict(self):
        """Get state dict for adapters only"""
        state_dict = {}
        for i, adapter in enumerate(self.visual_adapters):
            for name, param in adapter.named_parameters():
                state_dict[f'visual_adapters.{i}.{name}'] = param.data.clone()
        for i, adapter in enumerate(self.text_adapters):
            for name, param in adapter.named_parameters():
                state_dict[f'text_adapters.{i}.{name}'] = param.data.clone()
        return state_dict

    def load_adapter_state_dict(self, state_dict):
        """Load adapter state dict"""
        for i, adapter in enumerate(self.visual_adapters):
            for name, param in adapter.named_parameters():
                key = f'visual_adapters.{i}.{name}'
                if key in state_dict:
                    param.data.copy_(state_dict[key])
        for i, adapter in enumerate(self.text_adapters):
            for name, param in adapter.named_parameters():
                key = f'text_adapters.{i}.{name}'
                if key in state_dict:
                    param.data.copy_(state_dict[key])


class LoRAAdapter(nn.Module):
    """
    Low-Rank Adaptation (LoRA) adapter.

    Decomposes weight update into low-rank matrices: W' = W + BA
    where B is (d, r) and A is (r, k).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize A with Kaiming, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute low-rank update"""
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
