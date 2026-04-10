"""
Federated Learning Client
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Optional, Tuple, List
import copy

from .losses import RetrievalLoss


class FederatedClient:
    """
    Federated client for cross-modal retrieval.

    Handles local training, prototype computation, and update generation.
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataset: Dataset,
        modality_mask: Tuple[bool, bool],  # (has_image, has_text)
        config: dict,
        device: str = "cuda",
    ):
        self.client_id = client_id
        self.model = model
        self.dataset = dataset
        self.has_image, self.has_text = modality_mask
        self.config = config
        self.device = device

        # Training config
        self.batch_size = config.get('batch_size', 128)
        self.local_epochs = config.get('local_epochs', 1)
        self.lr = config.get('lr', 1e-4)
        self.weight_decay = config.get('weight_decay', 0.01)

        # Loss function
        self.criterion = RetrievalLoss(
            temperature=config.get('temperature', 0.07),
            lambda_align=config.get('lambda_align', 0.1),
            lambda_anchor=config.get('lambda_anchor', 1.0),
            lambda_stab=config.get('lambda_stab', 0.01),
            lambda_cma=config.get('lambda_cma', 0.0),
            lambda_clip_anchor=config.get('lambda_clip_anchor', 0.0),  # NEW
        )

        # Whether to use CLIP anchor loss
        self.use_clip_anchor = config.get('lambda_clip_anchor', 0.0) > 0

        # Global prototypes (received from server)
        self.global_image_proto = None
        self.global_text_proto = None

        # Local personalized state (for personalization)
        self.local_state = None

        # Statistics
        self.num_samples = len(dataset)

    def create_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Create dataloader for local dataset"""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,  # Avoid multiprocessing issues with CLIP tokenizer
            pin_memory=True,
            drop_last=True,
        )

    def receive_global_model(self, global_state: Dict[str, torch.Tensor]):
        """Receive global model from server"""
        self.model.load_trainable_state_dict(global_state)

    def receive_global_prototypes(
        self,
        image_proto: torch.Tensor,
        text_proto: torch.Tensor,
    ):
        """Receive global prototypes from server"""
        self.global_image_proto = image_proto.to(self.device)
        self.global_text_proto = text_proto.to(self.device)

    def local_train(self) -> Dict[str, float]:
        """
        Perform local training.

        Returns:
            Training statistics dict
        """
        self.model.train()

        # Get trainable parameters
        trainable_params = list(self.model.get_trainable_params().values())

        # Optimizer
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Store initial state for update computation
        initial_state = self.model.get_trainable_state_dict()

        dataloader = self.create_dataloader()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.local_epochs):
            for batch in dataloader:
                optimizer.zero_grad()

                # Get batch data
                images = batch['image'].to(self.device)
                texts = batch['text'].to(self.device)

                # Forward pass based on modality availability
                image_features = None
                text_features = None
                frozen_image_features = None
                frozen_text_features = None
                adapted_image_before_proj = None
                adapted_text_before_proj = None

                if self.has_image:
                    image_features = self.model.encode_image(images)
                    # Get features for CLIP anchor loss (before projection, same space as frozen)
                    if self.use_clip_anchor:
                        frozen_image_features = self.model.encode_image_frozen(images)
                        # Use before_proj features for comparison (same space as frozen)
                        adapted_image_before_proj = self.model.encode_image_before_proj(images)

                if self.has_text:
                    text_features = self.model.encode_text(texts)
                    if self.use_clip_anchor:
                        frozen_text_features = self.model.encode_text_frozen(texts)
                        adapted_text_before_proj = self.model.encode_text_before_proj(texts)

                # Compute loss
                loss, loss_dict = self.criterion(
                    image_features=image_features,
                    text_features=text_features,
                    has_image=self.has_image,
                    has_text=self.has_text,
                    global_image_proto=self.global_image_proto,
                    global_text_proto=self.global_text_proto,
                    frozen_image_features=frozen_image_features,
                    frozen_text_features=frozen_text_features,
                    adapted_image_before_proj=adapted_image_before_proj,
                    adapted_text_before_proj=adapted_text_before_proj,
                )

                # Backward pass
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        return {
            'loss': avg_loss,
            'num_samples': self.num_samples,
            'num_batches': num_batches,
        }

    def compute_update(self, initial_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute model update (delta)"""
        current_state = self.model.get_trainable_state_dict()
        update = {}
        for key in current_state:
            update[key] = current_state[key] - initial_state[key]
        return update

    @torch.no_grad()
    def compute_prototypes(self, num_samples: int = 128) -> Dict[str, torch.Tensor]:
        """
        Compute cross-modal prototypes from local data.

        Args:
            num_samples: Number of samples to use for prototype computation

        Returns:
            Dict with 'image_prototype' and 'text_prototype'
        """
        self.model.eval()

        dataloader = self.create_dataloader(shuffle=True)

        image_embeddings = []
        text_embeddings = []
        samples_collected = 0

        for batch in dataloader:
            if samples_collected >= num_samples:
                break

            images = batch['image'].to(self.device)
            texts = batch['text'].to(self.device)

            if self.has_image:
                img_emb = self.model.encode_image(images)
                image_embeddings.append(img_emb)

            if self.has_text:
                txt_emb = self.model.encode_text(texts)
                text_embeddings.append(txt_emb)

            samples_collected += images.shape[0]

        # Compute prototypes
        prototypes = {}

        if image_embeddings:
            all_img = torch.cat(image_embeddings, dim=0)
            prototypes['image_prototype'] = F.normalize(all_img.mean(dim=0), dim=-1)
        else:
            # Use global prototype as fallback for missing modality (better than zero)
            if self.global_image_proto is not None:
                prototypes['image_prototype'] = self.global_image_proto.detach().clone()
            else:
                # If no global prototype yet, use a small random vector to avoid zero issues
                prototypes['image_prototype'] = torch.randn(self.model.embed_dim, device=self.device)
                prototypes['image_prototype'] = F.normalize(prototypes['image_prototype'], dim=-1)

        if text_embeddings:
            all_txt = torch.cat(text_embeddings, dim=0)
            prototypes['text_prototype'] = F.normalize(all_txt.mean(dim=0), dim=-1)
        else:
            # Use global prototype as fallback for missing modality (better than zero)
            if self.global_text_proto is not None:
                prototypes['text_prototype'] = self.global_text_proto.detach().clone()
            else:
                # If no global prototype yet, use a small random vector to avoid zero issues
                prototypes['text_prototype'] = torch.randn(self.model.embed_dim, device=self.device)
                prototypes['text_prototype'] = F.normalize(prototypes['text_prototype'], dim=-1)

        return prototypes

    def compute_geometry(
        self,
        update: Dict[str, torch.Tensor],
        previous_update: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Compute update geometry features.

        Returns:
            Tensor of geometry features (norm, layer_ratio, cosine_to_prev, loss_scale)
        """
        device = self.device

        # Flatten update
        flat_update = torch.cat([v.flatten() for v in update.values()])

        # Feature 1: Update norm
        update_norm = flat_update.norm().item()

        # Feature 2: Layer ratio (visual vs text adapter ratio)
        visual_norm = 0.0
        text_norm = 0.0
        for key, val in update.items():
            if 'visual' in key:
                visual_norm += val.norm().item() ** 2
            elif 'text' in key:
                text_norm += val.norm().item() ** 2

        visual_norm = visual_norm ** 0.5
        text_norm = text_norm ** 0.5
        layer_ratio = visual_norm / (text_norm + 1e-8)

        # Feature 3: Cosine to previous update
        if previous_update is not None:
            flat_prev = torch.cat([v.flatten() for v in previous_update.values()])
            cosine_to_prev = F.cosine_similarity(
                flat_update.unsqueeze(0),
                flat_prev.unsqueeze(0)
            ).item()
        else:
            cosine_to_prev = 0.0

        # Feature 4: Normalized update magnitude
        loss_scale = update_norm / (self.num_samples ** 0.5 + 1e-8)

        return torch.tensor([update_norm, layer_ratio, cosine_to_prev, loss_scale], device=device)

    def get_client_stats(
        self,
        update: Dict[str, torch.Tensor],
        loss: float,
        previous_update: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get complete client statistics for router.

        Returns:
            Dict with prototypes, geometry, mask, and loss
        """
        prototypes = self.compute_prototypes()
        geometry = self.compute_geometry(update, previous_update)

        return {
            'image_prototype': prototypes['image_prototype'],
            'text_prototype': prototypes['text_prototype'],
            'geometry': geometry,
            'modality_mask': torch.tensor([self.has_image, self.has_text], dtype=torch.float32),
            'loss': torch.tensor(loss),
        }

    @torch.no_grad()
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Evaluate model on local or provided dataset.

        Returns:
            Evaluation metrics dict
        """
        self.model.eval()

        dataset = eval_dataset if eval_dataset is not None else self.dataset
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        total_loss = 0.0
        num_batches = 0

        image_embeddings = []
        text_embeddings = []

        for batch in dataloader:
            images = batch['image'].to(self.device)
            texts = batch['text'].to(self.device)

            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(texts)

            image_embeddings.append(image_features)
            text_embeddings.append(text_features)

            # Compute loss
            loss, _ = self.criterion(
                image_features=image_features,
                text_features=text_features,
                has_image=True,
                has_text=True,
            )
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Compute retrieval metrics
        all_img = torch.cat(image_embeddings, dim=0)
        all_txt = torch.cat(text_embeddings, dim=0)

        # Similarity matrix
        sims = all_img @ all_txt.T

        # Recall@K
        metrics = {'loss': avg_loss}
        for k in [1, 5, 10]:
            # Image-to-text
            i2t_topk = sims.topk(k, dim=1).indices
            i2t_correct = (i2t_topk == torch.arange(len(all_img), device=self.device).unsqueeze(1)).any(dim=1)
            metrics[f'i2t_r@{k}'] = i2t_correct.float().mean().item()

            # Text-to-image
            t2i_topk = sims.T.topk(k, dim=1).indices
            t2i_correct = (t2i_topk == torch.arange(len(all_txt), device=self.device).unsqueeze(1)).any(dim=1)
            metrics[f't2i_r@{k}'] = t2i_correct.float().mean().item()

        return metrics
