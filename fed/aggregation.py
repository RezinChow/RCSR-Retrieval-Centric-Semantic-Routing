"""
Aggregation utilities for federated learning
"""

import torch
from typing import Dict, List, Optional
import copy


def aggregate_updates(
    updates: List[Dict[str, torch.Tensor]],
    weights: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Aggregate model updates using weighted average.

    Args:
        updates: List of client updates (each is a state dict)
        weights: Aggregation weights (should sum to 1)

    Returns:
        Aggregated update state dict
    """
    if len(updates) == 0:
        raise ValueError("No updates to aggregate")

    # Initialize with zeros
    aggregated = {}
    for key in updates[0]:
        aggregated[key] = torch.zeros_like(updates[0][key])

    # Weighted sum
    for update, weight in zip(updates, weights):
        for key in aggregated:
            aggregated[key] = aggregated[key] + weight * update[key]

    return aggregated


def fedavg_aggregate(
    states: List[Dict[str, torch.Tensor]],
    num_samples: List[int],
) -> Dict[str, torch.Tensor]:
    """
    FedAvg aggregation weighted by number of samples.

    Args:
        states: List of client model states
        num_samples: Number of samples per client

    Returns:
        Aggregated state dict
    """
    total_samples = sum(num_samples)
    weights = torch.tensor([n / total_samples for n in num_samples])

    return aggregate_updates(states, weights)


def fedprox_aggregate(
    states: List[Dict[str, torch.Tensor]],
    num_samples: List[int],
    global_state: Dict[str, torch.Tensor],
    mu: float = 0.01,
) -> Dict[str, torch.Tensor]:
    """
    FedProx-style aggregation (same as FedAvg, proximal term is in local training).

    Args:
        states: List of client model states
        num_samples: Number of samples per client
        global_state: Current global model state
        mu: Proximal regularization coefficient (used in local training)

    Returns:
        Aggregated state dict
    """
    # FedProx aggregation is same as FedAvg
    # The proximal term is applied during local training
    return fedavg_aggregate(states, num_samples)


def compute_update_geometry(
    update: Dict[str, torch.Tensor],
    previous_update: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    Compute geometry features of a model update.

    Args:
        update: Model update (delta)
        previous_update: Optional previous update for cosine computation

    Returns:
        Dict with geometry features
    """
    # Flatten update
    flat_update = torch.cat([v.flatten().float() for v in update.values()])

    # Update norm
    update_norm = flat_update.norm().item()

    # Separate visual and text adapter norms
    visual_params = []
    text_params = []
    proj_params = []

    for key, val in update.items():
        if 'visual' in key.lower():
            visual_params.append(val.flatten())
        elif 'text' in key.lower():
            text_params.append(val.flatten())
        elif 'proj' in key.lower():
            proj_params.append(val.flatten())

    visual_norm = torch.cat(visual_params).norm().item() if visual_params else 0.0
    text_norm = torch.cat(text_params).norm().item() if text_params else 0.0
    proj_norm = torch.cat(proj_params).norm().item() if proj_params else 0.0

    # Layer ratio
    layer_ratio = visual_norm / (text_norm + 1e-8)

    # Cosine to previous update
    cosine_to_prev = 0.0
    if previous_update is not None:
        flat_prev = torch.cat([v.flatten().float() for v in previous_update.values()])
        if flat_prev.norm() > 0 and flat_update.norm() > 0:
            cosine_to_prev = (flat_update @ flat_prev / (flat_update.norm() * flat_prev.norm())).item()

    return {
        'update_norm': update_norm,
        'visual_norm': visual_norm,
        'text_norm': text_norm,
        'proj_norm': proj_norm,
        'layer_ratio': layer_ratio,
        'cosine_to_prev': cosine_to_prev,
    }


def clip_update(
    update: Dict[str, torch.Tensor],
    max_norm: float,
) -> Dict[str, torch.Tensor]:
    """
    Clip update norm for privacy/stability.

    Args:
        update: Model update
        max_norm: Maximum allowed norm

    Returns:
        Clipped update
    """
    # Flatten update
    flat_update = torch.cat([v.flatten() for v in update.values()])
    current_norm = flat_update.norm()

    if current_norm > max_norm:
        scale = max_norm / (current_norm + 1e-8)
        clipped = {}
        for key, val in update.items():
            clipped[key] = val * scale
        return clipped

    return update


def compute_update_similarity(
    update1: Dict[str, torch.Tensor],
    update2: Dict[str, torch.Tensor],
) -> float:
    """
    Compute cosine similarity between two updates.

    Args:
        update1: First update
        update2: Second update

    Returns:
        Cosine similarity
    """
    flat1 = torch.cat([v.flatten().float() for v in update1.values()])
    flat2 = torch.cat([v.flatten().float() for v in update2.values()])

    if flat1.norm() == 0 or flat2.norm() == 0:
        return 0.0

    return (flat1 @ flat2 / (flat1.norm() * flat2.norm())).item()
