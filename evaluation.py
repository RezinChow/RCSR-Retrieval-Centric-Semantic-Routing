"""
Evaluation utilities for cross-modal retrieval
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict
from tqdm import tqdm


def evaluate_retrieval(
    model,
    dataset: Dataset,
    batch_size: int = 128,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Evaluate cross-modal retrieval performance.
    
    Args:
        model: The model to evaluate
        dataset: Test dataset
        batch_size: Batch size for evaluation
        device: Device to use
        
    Returns:
        Dict with retrieval metrics
    """
    model.eval()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    all_image_features = []
    all_text_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['image'].to(device)
            texts = batch['text'].to(device)
            
            # Encode
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)
            
            # Normalize
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            all_image_features.append(image_features.cpu())
            all_text_features.append(text_features.cpu())
    
    # Concatenate all features
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)
    
    # Compute similarity matrix
    similarity_matrix = all_image_features @ all_text_features.T
    
    # Compute metrics
    metrics = {}
    
    # Image-to-Text retrieval
    i2t_ranks = []
    for i in range(len(all_image_features)):
        # Get similarity scores for this image against all texts
        sims = similarity_matrix[i]
        # Rank texts by similarity
        ranked_indices = torch.argsort(sims, descending=True)
        # Find rank of correct text (diagonal)
        rank = (ranked_indices == i).nonzero(as_tuple=True)[0].item()
        i2t_ranks.append(rank)
    
    i2t_ranks = torch.tensor(i2t_ranks)
    
    # Text-to-Image retrieval
    t2i_ranks = []
    for i in range(len(all_text_features)):
        # Get similarity scores for this text against all images
        sims = similarity_matrix[:, i]
        # Rank images by similarity
        ranked_indices = torch.argsort(sims, descending=True)
        # Find rank of correct image (diagonal)
        rank = (ranked_indices == i).nonzero(as_tuple=True)[0].item()
        t2i_ranks.append(rank)
    
    t2i_ranks = torch.tensor(t2i_ranks)
    
    # Compute Recall@K
    for k in [1, 5, 10]:
        metrics[f'i2t_r@{k}'] = (i2t_ranks < k).float().mean().item() * 100
        metrics[f't2i_r@{k}'] = (t2i_ranks < k).float().mean().item() * 100
    
    # Mean recall
    metrics['mean_r@1'] = (metrics['i2t_r@1'] + metrics['t2i_r@1']) / 2
    metrics['mean_r@5'] = (metrics['i2t_r@5'] + metrics['t2i_r@5']) / 2
    metrics['mean_r@10'] = (metrics['i2t_r@10'] + metrics['t2i_r@10']) / 2
    
    # Median rank
    metrics['i2t_median_rank'] = i2t_ranks.float().median().item()
    metrics['t2i_median_rank'] = t2i_ranks.float().median().item()
    
    return metrics
