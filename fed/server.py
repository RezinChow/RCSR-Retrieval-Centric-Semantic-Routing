"""
Federated Learning Server with RCSR Router
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import copy

from models.router import RCSRRouter, MinimaxGameWeighter
from .aggregation import aggregate_updates


class FederatedServer:
    """
    Federated server with Retrieval-Centric Semantic Routing (RCSR).

    Manages global model, router, and minimax game weighting.
    """

    def __init__(
        self,
        model: nn.Module,
        num_clients: int,
        config: dict,
        device: str = "cuda",
    ):
        self.model = model
        self.num_clients = num_clients
        self.config = config
        self.device = device

        # Global model state
        self.global_state = model.get_trainable_state_dict()

        # Router
        router_config = config.get('router', {})
        self.use_router = router_config.get('enabled', True)

        if self.use_router:
            self.router = RCSRRouter(
                embed_dim=config['model'].get('embed_dim', 512),
                hidden_dim=router_config.get('hidden_dim', 128),
                geometry_dim=4,
                num_layers=router_config.get('num_layers', 2),
                use_attention=router_config.get('use_attention', True),
                beta1=router_config.get('beta1', 1.0),
                beta2=router_config.get('beta2', 1.0),
                beta4=router_config.get('beta4', 0.1),
                beta5=router_config.get('beta5', 0.3),
            ).to(device)

            self.router_optimizer = torch.optim.Adam(
                self.router.parameters(),
                lr=router_config.get('lr', 1e-3),
            )
        else:
            self.router = None

        # Minimax game weighter
        minimax_config = config.get('minimax', {})
        self.use_minimax = minimax_config.get('enabled', True)

        if self.use_minimax:
            self.minimax = MinimaxGameWeighter(
                num_clients=num_clients,
                tau=minimax_config.get('tau', 0.1),
                eta_q=minimax_config.get('eta_q', 0.1),
            )
        else:
            self.minimax = None

        # Global prototypes
        embed_dim = config['model'].get('embed_dim', 512)
        self.global_image_proto = torch.zeros(embed_dim, device=device)
        self.global_text_proto = torch.zeros(embed_dim, device=device)
        self.ema_decay = config.get('prototype', {}).get('ema_decay', 0.95)

        # Statistics
        self.round_num = 0
        self.history = {
            'router_loss': [],
            'weights_entropy': [],
            'cross_modal_coherence': [],
        }

    def select_clients(self, participation_rate: float) -> List[int]:
        """Randomly select clients for this round"""
        num_selected = max(1, int(self.num_clients * participation_rate))
        selected = np.random.choice(
            self.num_clients,
            size=num_selected,
            replace=False
        ).tolist()
        return selected

    def broadcast_global_model(self) -> Dict[str, torch.Tensor]:
        """Get global model state for broadcasting"""
        return {k: v.clone().detach() for k, v in self.global_state.items()}

    def broadcast_global_prototypes(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get global prototypes for broadcasting"""
        return (
            self.global_image_proto.clone(),
            self.global_text_proto.clone(),
        )

    def _compute_align_direction(self) -> torch.Tensor:
        """计算全局对齐方向"""
        if self.global_image_proto.abs().sum() < 1e-8 or self.global_text_proto.abs().sum() < 1e-8:
            # 如果任一原型为零，返回默认对齐方向
            align = self.global_image_proto - self.global_text_proto
        else:
            align = self.global_image_proto - self.global_text_proto
        return F.normalize(align, dim=-1)

    def aggregate(
        self,
        client_updates: Dict[int, Dict[str, torch.Tensor]],
        client_stats: Dict[int, Dict[str, torch.Tensor]],
        client_losses: Dict[int, float],
    ) -> Dict[str, float]:
        """
        Aggregate client updates using RCSR + minimax weights.

        Args:
            client_updates: Dict mapping client_id -> model update
            client_stats: Dict mapping client_id -> client statistics
            client_losses: Dict mapping client_id -> scalar loss

        Returns:
            Aggregation statistics dict
        """
        selected_clients = list(client_updates.keys())
        num_selected = len(selected_clients)

        # Prepare client stats list (ordered)
        stats_list = [client_stats[cid] for cid in selected_clients]

        # Step 1: Compute router weights (before any prototype updates)
        if self.use_router and self.router is not None:
            router_weights = self.router(stats_list)
        else:
            # Uniform weights
            router_weights = torch.ones(num_selected, device=self.device) / num_selected

        # Step 2: Compute minimax weights
        if self.use_minimax and self.minimax is not None:
            # Update minimax based on losses
            self.minimax.update(client_losses, selected_clients)
            minimax_weights = self.minimax.get_weights(selected_clients).to(self.device)
        else:
            minimax_weights = torch.ones(num_selected, device=self.device) / num_selected

        # Step 3: Fuse weights
        fused_weights = router_weights * minimax_weights
        fused_weights = fused_weights / fused_weights.sum()

        # Step 4: Update server global prototypes FIRST (before router uses them)
        self._update_global_prototypes(stats_list, fused_weights.detach())

        # Step 5: Sync router global prototypes from server (CRITICAL FIX)
        if self.use_router and self.router is not None:
            self.router.global_image_proto = self.global_image_proto.detach().clone()
            self.router.global_text_proto = self.global_text_proto.detach().clone()
            self.router.global_align_direction = self._compute_align_direction().detach().clone()

        # Step 6: Aggregate updates
        updates_list = [client_updates[cid] for cid in selected_clients]
        aggregated_update = aggregate_updates(updates_list, fused_weights)

        # Step 7: Update global model
        for key in self.global_state:
            self.global_state[key] = self.global_state[key] + aggregated_update[key]

        # Load aggregated state into model
        self.model.load_trainable_state_dict(self.global_state)

        # Step 8: Train router (if enabled) - uses already synced prototypes
        router_loss = 0.0
        if self.use_router and self.router is not None:
            self.router_optimizer.zero_grad()
            loss, loss_dict = self.router.compute_loss(router_weights, stats_list)
            loss.backward()
            self.router_optimizer.step()
            router_loss = loss.item()

        # Step 9: Update server global prototypes for next round (after aggregation)
        # This will be synced to router at the start of next round
        self._update_global_prototypes(stats_list, fused_weights.detach())

        # Compute statistics
        stats = {
            'router_loss': router_loss,
            'weights_entropy': -(fused_weights * torch.log(fused_weights + 1e-10)).sum().item(),
            'max_weight': fused_weights.max().item(),
            'min_weight': fused_weights.min().item(),
        }

        # Router diagnostics
        if self.use_router and self.router is not None:
            diagnostics = self.router.get_diagnostics(router_weights)
            stats.update(diagnostics)

        # Update history
        self.history['router_loss'].append(stats['router_loss'])
        self.history['weights_entropy'].append(stats['weights_entropy'])
        if 'cross_modal_coherence' in stats:
            self.history['cross_modal_coherence'].append(stats['cross_modal_coherence'])

        self.round_num += 1

        return stats

    def _update_global_prototypes(
        self,
        client_stats: List[Dict[str, torch.Tensor]],
        weights: torch.Tensor,
    ):
        """Update global prototypes using EMA"""
        # Stack prototypes
        image_protos = torch.stack([s['image_prototype'] for s in client_stats]).to(self.device)
        text_protos = torch.stack([s['text_prototype'] for s in client_stats]).to(self.device)

        # Compute weighted aggregated prototypes
        agg_image = (weights.unsqueeze(-1) * image_protos).sum(dim=0)
        agg_text = (weights.unsqueeze(-1) * text_protos).sum(dim=0)

        # EMA update
        self.global_image_proto = (
            self.ema_decay * self.global_image_proto +
            (1 - self.ema_decay) * agg_image
        )
        self.global_text_proto = (
            self.ema_decay * self.global_text_proto +
            (1 - self.ema_decay) * agg_text
        )

        # Safe normalize (handles zero vectors)
        self.global_image_proto = self._safe_normalize(self.global_image_proto)
        self.global_text_proto = self._safe_normalize(self.global_text_proto)

    def _safe_normalize(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Safe normalization that handles zero vectors"""
        norm = x.norm(dim=-1, keepdim=True)
        return torch.where(norm > eps, x / norm, torch.zeros_like(x))

    def get_global_model(self) -> nn.Module:
        """Get global model with current state"""
        return self.model

    def get_client_weights_history(self) -> Dict[str, List[float]]:
        """Get history of weight statistics"""
        return self.history

    def save_checkpoint(self, path: str):
        """Save server checkpoint"""
        checkpoint = {
            'round_num': self.round_num,
            'global_state': self.global_state,
            'global_image_proto': self.global_image_proto,
            'global_text_proto': self.global_text_proto,
            'history': self.history,
        }

        if self.router is not None:
            checkpoint['router_state'] = self.router.state_dict()

        if self.minimax is not None:
            checkpoint['minimax_q'] = self.minimax.q

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load server checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.round_num = checkpoint['round_num']
        self.global_state = checkpoint['global_state']
        self.global_image_proto = checkpoint['global_image_proto']
        self.global_text_proto = checkpoint['global_text_proto']
        self.history = checkpoint['history']

        if self.router is not None and 'router_state' in checkpoint:
            self.router.load_state_dict(checkpoint['router_state'])

        if self.minimax is not None and 'minimax_q' in checkpoint:
            self.minimax.q = checkpoint['minimax_q']

        # Load global state into model
        self.model.load_trainable_state_dict(self.global_state)
