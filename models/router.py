"""
Retrieval-Centric Semantic Router (RCSR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


def safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Safe normalization that handles zero vectors.

    Args:
        x: Tensor to normalize
        eps: Small epsilon to avoid division by zero

    Returns:
        Normalized tensor, or zero tensor if input is near-zero
    """
    norm = x.norm(dim=-1, keepdim=True)
    return torch.where(norm > eps, x / norm, torch.zeros_like(x))


class ClientTokenEncoder(nn.Module):
    """
    Encode client statistics into a fixed-dimensional token.

    Input features:
        - Image prototype (embed_dim)
        - Text prototype (embed_dim)
        - Cross-modal similarity (1)
        - Update geometry (variable)
        - Modality mask (2)
        - Scalar loss (1)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        geometry_dim: int = 4,  # norm, layer_ratio, cosine_to_prev, etc.
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.geometry_dim = geometry_dim

        # Input dimension: 2*embed_dim + 1 + geometry_dim + 2 + 1
        input_dim = 2 * embed_dim + 1 + geometry_dim + 2 + 1

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        image_prototype: torch.Tensor,       # (B, embed_dim)
        text_prototype: torch.Tensor,        # (B, embed_dim)
        cross_modal_sim: torch.Tensor,       # (B, 1)
        geometry: torch.Tensor,              # (B, geometry_dim)
        modality_mask: torch.Tensor,         # (B, 2)
        scalar_loss: torch.Tensor,           # (B, 1)
    ) -> torch.Tensor:
        """
        Encode client statistics into token.

        Args:
            image_prototype: L2-normalized image prototype
            text_prototype: L2-normalized text prototype
            cross_modal_sim: Cosine similarity between prototypes
            geometry: Update geometry features
            modality_mask: (has_image, has_text) binary mask
            scalar_loss: Validation loss scalar

        Returns:
            Client token of shape (B, hidden_dim)
        """
        # Concatenate all features
        features = torch.cat([
            image_prototype,
            text_prototype,
            cross_modal_sim,
            geometry,
            modality_mask.float(),
            scalar_loss,
        ], dim=-1)

        return self.encoder(features)


class SelfAttentionBlock(nn.Module):
    """Self-attention block for client interaction"""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tokens (1, num_clients, hidden_dim)
            mask: Optional attention mask

        Returns:
            Output tokens of same shape
        """
        # Self-attention
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class RCSRRouter(nn.Module):
    """
    Retrieval-Centric Semantic Router (RCSR) with Alignment Consistency.

    Learns aggregation weights from cross-modal prototypes, update geometry,
    and modality masks, optimized for retrieval-space consistency.

    NEW: Alignment consistency constraint to prevent alignment direction conflicts
    during aggregation.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 128,
        geometry_dim: int = 4,
        num_layers: int = 2,
        use_attention: bool = True,
        num_heads: int = 4,
        dropout: float = 0.1,
        # Loss weights
        beta1: float = 1.0,  # global image prototype consistency
        beta2: float = 1.0,  # global text prototype consistency
        beta4: float = 0.1,  # entropy regularization
        beta5: float = 0.3,  # alignment consistency constraint
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        # Loss weights
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta4 = beta4
        self.beta5 = beta5

        # Client token encoder
        self.token_encoder = ClientTokenEncoder(
            embed_dim=embed_dim,
            geometry_dim=geometry_dim,
            hidden_dim=hidden_dim,
        )

        # Optional self-attention for client interaction
        if use_attention:
            self.attention_blocks = nn.ModuleList([
                SelfAttentionBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_layers)
            ])
        else:
            # MLP-only version
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # Weight output head
        self.weight_head = nn.Linear(hidden_dim, 1)

        # Global prototype EMA (stored as buffers)
        self.register_buffer('global_image_proto', torch.zeros(embed_dim))
        self.register_buffer('global_text_proto', torch.zeros(embed_dim))
        # NEW: Global alignment direction (image_proto - text_proto normalized)
        self.register_buffer('global_align_direction', torch.zeros(embed_dim))
        self.ema_decay = 0.95

    def forward(
        self,
        client_stats: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Compute router weights for selected clients.

        Args:
            client_stats: List of client statistics dicts with keys:
                - 'image_prototype': (embed_dim,)
                - 'text_prototype': (embed_dim,)
                - 'geometry': (geometry_dim,)
                - 'modality_mask': (2,)
                - 'loss': scalar

        Returns:
            Router weights of shape (num_clients,), sum to 1
        """
        num_clients = len(client_stats)
        device = next(self.parameters()).device

        # Stack client features
        image_protos = torch.stack([s['image_prototype'] for s in client_stats]).to(device)
        text_protos = torch.stack([s['text_prototype'] for s in client_stats]).to(device)
        geometries = torch.stack([s['geometry'] for s in client_stats]).to(device)
        masks = torch.stack([s['modality_mask'] for s in client_stats]).to(device)
        losses = torch.stack([s['loss'] for s in client_stats]).unsqueeze(-1).to(device)

        # Compute cross-modal similarities
        cross_sims = F.cosine_similarity(image_protos, text_protos, dim=-1).unsqueeze(-1)

        # Encode client tokens
        tokens = self.token_encoder(
            image_protos, text_protos, cross_sims, geometries, masks, losses
        )  # (num_clients, hidden_dim)

        # Apply attention or MLP
        if self.use_attention:
            tokens = tokens.unsqueeze(0)  # (1, num_clients, hidden_dim)
            for attn_block in self.attention_blocks:
                tokens = attn_block(tokens)
            tokens = tokens.squeeze(0)  # (num_clients, hidden_dim)
        else:
            tokens = self.mlp(tokens)

        # Compute weights
        logits = self.weight_head(tokens).squeeze(-1)  # (num_clients,)
        weights = F.softmax(logits, dim=0)

        return weights

    def compute_alignment_directions(
        self,
        image_protos: torch.Tensor,
        text_protos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute alignment directions for each client.

        Alignment direction represents how each client aligns image and text spaces.
        Clients with similar alignment directions should be aggregated together.

        Args:
            image_protos: (num_clients, embed_dim)
            text_protos: (num_clients, embed_dim)

        Returns:
            Alignment directions of shape (num_clients, embed_dim)
        """
        # Alignment direction = normalized difference between image and text protos
        align_dirs = image_protos - text_protos
        align_dirs = F.normalize(align_dirs, dim=-1)
        return align_dirs

    def compute_alignment_consistency(
        self,
        weights: torch.Tensor,
        align_dirs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute alignment consistency metrics.

        Args:
            weights: Router weights (num_clients,)
            align_dirs: Alignment directions (num_clients, embed_dim)

        Returns:
            Tuple of (weighted_align_dir, consistency_score)
        """
        # Weighted aggregated alignment direction
        weighted_align = (weights.unsqueeze(-1) * align_dirs).sum(dim=0)
        weighted_align = safe_normalize(weighted_align)

        # Consistency: how aligned each client is with the weighted direction
        # Higher consistency = client aligns well with aggregated direction
        consistencies = F.cosine_similarity(align_dirs, weighted_align.unsqueeze(0), dim=-1)

        return weighted_align, consistencies

    def compute_loss(
        self,
        weights: torch.Tensor,
        client_stats: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute router training loss based on retrieval-space consistency.

        Loss = beta1 * (1 - cos(g_hat_I, g_I))
             + beta2 * (1 - cos(g_hat_T, g_T))
             + beta4 * H(weights)
             + beta5 * alignment_consistency_loss

        Args:
            weights: Router weights (num_clients,)
            client_stats: Client statistics

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        device = weights.device

        # Stack prototypes
        image_protos = torch.stack([s['image_prototype'] for s in client_stats]).to(device)
        text_protos = torch.stack([s['text_prototype'] for s in client_stats]).to(device)

        # Compute weighted aggregated prototypes
        g_hat_I = (weights.unsqueeze(-1) * image_protos).sum(dim=0)
        g_hat_T = (weights.unsqueeze(-1) * text_protos).sum(dim=0)

        # Normalize (use safe_normalize to handle zero vectors)
        g_hat_I = safe_normalize(g_hat_I)
        g_hat_T = safe_normalize(g_hat_T)

        # Get global prototypes (EMA targets) - use safe_normalize
        g_I = safe_normalize(self.global_image_proto)
        g_T = safe_normalize(self.global_text_proto)

        # Compute losses
        loss_img_consist = 1 - F.cosine_similarity(g_hat_I.unsqueeze(0), g_I.unsqueeze(0))
        loss_txt_consist = 1 - F.cosine_similarity(g_hat_T.unsqueeze(0), g_T.unsqueeze(0))

        # Entropy regularization (encourage diverse weights)
        entropy = -(weights * torch.log(weights + 1e-10)).sum()
        loss_entropy = -entropy  # Maximize entropy

        # Alignment consistency loss
        align_dirs = self.compute_alignment_directions(image_protos, text_protos)
        weighted_align, consistencies = self.compute_alignment_consistency(weights, align_dirs)

        # Loss 1: Aggregated alignment should be consistent with global alignment
        g_align = safe_normalize(self.global_align_direction)
        loss_align_global = 1 - F.cosine_similarity(weighted_align.unsqueeze(0), g_align.unsqueeze(0))

        # Loss 2: Penalize giving high weights to clients with conflicting alignments
        loss_align_conflict = (weights * (1 - consistencies)).sum()

        # Combined alignment consistency loss
        loss_align_consist = 0.5 * loss_align_global + 0.5 * loss_align_conflict

        # Total loss - use softplus to ensure positive loss
        total_loss = (
            self.beta1 * loss_img_consist +
            self.beta2 * loss_txt_consist +
            self.beta4 * loss_entropy +
            self.beta5 * loss_align_consist
        )
        # Ensure loss is positive for stable training
        total_loss = F.softplus(total_loss)

        loss_dict = {
            'loss_img_consist': loss_img_consist.item(),
            'loss_txt_consist': loss_txt_consist.item(),
            'loss_entropy': loss_entropy.item(),
            'weight_entropy': entropy.item(),
            'loss_align_consist': loss_align_consist.item(),
            'align_consistency_mean': consistencies.mean().item(),
        }

        return total_loss.squeeze(), loss_dict

    @torch.no_grad()
    def update_global_prototypes(
        self,
        weights: torch.Tensor,
        client_stats: List[Dict[str, torch.Tensor]],
    ):
        """Update global prototypes and alignment direction using EMA"""
        device = self.global_image_proto.device

        # Stack prototypes
        image_protos = torch.stack([s['image_prototype'] for s in client_stats]).to(device)
        text_protos = torch.stack([s['text_prototype'] for s in client_stats]).to(device)

        # Compute weighted aggregated prototypes
        g_hat_I = (weights.unsqueeze(-1) * image_protos).sum(dim=0)
        g_hat_T = (weights.unsqueeze(-1) * text_protos).sum(dim=0)

        # NEW: Compute weighted alignment direction
        align_dirs = self.compute_alignment_directions(image_protos, text_protos)
        g_hat_align = (weights.unsqueeze(-1) * align_dirs).sum(dim=0)
        g_hat_align = safe_normalize(g_hat_align)

        # EMA update
        self.global_image_proto = (
            self.ema_decay * self.global_image_proto +
            (1 - self.ema_decay) * g_hat_I
        )
        self.global_text_proto = (
            self.ema_decay * self.global_text_proto +
            (1 - self.ema_decay) * g_hat_T
        )
        # NEW: Update global alignment direction
        self.global_align_direction = (
            self.ema_decay * self.global_align_direction +
            (1 - self.ema_decay) * g_hat_align
        )

    def get_diagnostics(self, weights: torch.Tensor) -> Dict[str, float]:
        """Get router diagnostics"""
        entropy = -(weights * torch.log(weights + 1e-10)).sum().item()
        max_weight = weights.max().item()
        min_weight = weights.min().item()

        # Cross-modal coherence
        g_I = safe_normalize(self.global_image_proto)
        g_T = safe_normalize(self.global_text_proto)
        coherence = F.cosine_similarity(g_I.unsqueeze(0), g_T.unsqueeze(0)).item()

        # NEW: Alignment direction magnitude (indicates alignment stability)
        align_magnitude = self.global_align_direction.norm().item()

        return {
            'weight_entropy': entropy,
            'max_weight': max_weight,
            'min_weight': min_weight,
            'cross_modal_coherence': coherence,
            'align_direction_magnitude': align_magnitude,  # NEW
        }


class MinimaxGameWeighter:
    """
    Robust-fair mixture game for worst-case reweighting.

    Maintains adversarial mixture q over clients and updates via
    exponentiated gradient descent on client losses.
    """

    def __init__(
        self,
        num_clients: int,
        tau: float = 0.1,      # entropy regularization
        eta_q: float = 0.1,   # learning rate for q
    ):
        self.num_clients = num_clients
        self.tau = tau
        self.eta_q = eta_q

        # Initialize uniform mixture
        self.q = torch.ones(num_clients) / num_clients

    def update(self, client_losses: Dict[int, float], selected_clients: List[int]):
        """
        Update mixture weights using exponentiated gradient.

        q_k <- q_k * exp(eta_q * loss_k) / Z
        """
        # Only update for selected clients
        for client_id in selected_clients:
            if client_id in client_losses:
                loss = client_losses[client_id]
                self.q[client_id] *= math.exp(self.eta_q * loss)

        # Add entropy regularization and normalize
        self.q = self.q ** (1 / (1 + self.tau))
        self.q = self.q / self.q.sum()

    def get_weights(self, selected_clients: List[int]) -> torch.Tensor:
        """Get normalized weights for selected clients"""
        weights = self.q[selected_clients]
        return weights / weights.sum()

    def reset(self):
        """Reset to uniform distribution"""
        self.q = torch.ones(self.num_clients) / self.num_clients

    def get_statistics(self) -> Dict[str, float]:
        """Get mixture statistics"""
        return {
            'q_max': self.q.max().item(),
            'q_min': self.q.min().item(),
            'q_entropy': -(self.q * torch.log(self.q + 1e-10)).sum().item(),
        }
