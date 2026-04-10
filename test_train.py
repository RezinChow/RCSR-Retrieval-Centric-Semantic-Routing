"""
Simplified training test script
"""
import os
import torch
import clip
from tqdm import tqdm

# Set environment to avoid memory issues
os.environ['OPENBLAS_NUM_THREADS'] = '4'

from data.dataset import Flickr30kDataset
from data.partition import create_federated_partition, FederatedDataset
from models.clip_model import create_clip_model
from fed.losses import InfoNCELoss
from torch.utils.data import DataLoader, Subset

def main():
    print("=" * 50)
    print("RCSR Quick Training Test")
    print("=" * 50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load CLIP preprocessing
    _, preprocess = clip.load("ViT-B/32", device=device)

    # Load dataset (use subset for quick test)
    print("\n[1] Loading dataset...")
    full_dataset = Flickr30kDataset(
        data_root="./datasets/flickr30k",
        split="train",
        transform=preprocess,
    )

    # Use only 2000 samples for quick test
    subset_size = min(2000, len(full_dataset))
    indices = list(range(subset_size))
    dataset = Subset(full_dataset, indices)
    print(f"Using {len(dataset)} samples for quick test")

    # Create federated partition (5 clients for quick test)
    print("\n[2] Creating federated partition...")
    num_clients = 5

    # Simple partition (even split)
    samples_per_client = len(dataset) // num_clients
    client_indices = {}
    for i in range(num_clients):
        start = i * samples_per_client
        end = start + samples_per_client
        client_indices[i] = list(range(start, end))

    # No missing modality for simplicity
    modality_masks = {i: (True, True) for i in range(num_clients)}

    print(f"Clients: {num_clients}, Samples/client: {samples_per_client}")

    # Create model
    print("\n[3] Creating model...")
    model = create_clip_model(
        backbone="ViT-B/32",
        embed_dim=512,
        adapter_dim=64,
        freeze_backbone=True,
        use_adapters=True,
        device=device,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Loss function
    criterion = InfoNCELoss(temperature=0.07)

    # Optimizer (only trainable params)
    trainable = list(model.get_trainable_params().values())
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)

    # Training loop (3 rounds, 1 epoch each)
    print("\n[4] Training...")
    num_rounds = 3

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num}/{num_rounds} ---")

        # Select clients (all for quick test)
        selected_clients = list(range(num_clients))

        round_loss = 0
        for client_id in selected_clients:
            # Get client data
            client_data = Subset(dataset, client_indices[client_id])
            dataloader = DataLoader(
                client_data,
                batch_size=32,
                shuffle=True,
                num_workers=4,  # 并行数据加载
                drop_last=True,
                persistent_workers=True,
            )

            # Local training
            model.train()
            client_loss = 0
            num_batches = 0

            for batch in dataloader:
                images = batch['image'].to(device)
                texts = batch['text'].to(device)

                optimizer.zero_grad()

                # Forward
                img_features = model.encode_image(images)
                txt_features = model.encode_text(texts)

                # Loss
                loss = criterion(img_features, txt_features)

                # Backward
                loss.backward()
                optimizer.step()

                client_loss += loss.item()
                num_batches += 1

            avg_loss = client_loss / max(num_batches, 1)
            round_loss += avg_loss
            print(f"  Client {client_id}: loss={avg_loss:.4f}")

        avg_round_loss = round_loss / len(selected_clients)
        print(f"  Round avg loss: {avg_round_loss:.4f}")

    # Quick evaluation
    print("\n[5] Evaluation...")
    model.eval()

    # Use test split (small subset)
    test_dataset = Flickr30kDataset(
        data_root="./datasets/flickr30k",
        split="test",
        transform=preprocess,
    )
    test_subset = Subset(test_dataset, list(range(min(500, len(test_dataset)))))
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=4)

    all_img_features = []
    all_txt_features = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            texts = batch['text'].to(device)

            img_feat = model.encode_image(images)
            txt_feat = model.encode_text(texts)

            all_img_features.append(img_feat)
            all_txt_features.append(txt_feat)

    img_features = torch.cat(all_img_features, dim=0)
    txt_features = torch.cat(all_txt_features, dim=0)

    # Compute recall@K
    sims = img_features @ txt_features.T
    num_samples = sims.shape[0]
    gt = torch.arange(num_samples, device=device)

    for k in [1, 5, 10]:
        # Image-to-text
        i2t_topk = sims.topk(k, dim=1).indices
        i2t_correct = (i2t_topk == gt.unsqueeze(1)).any(dim=1)
        i2t_recall = i2t_correct.float().mean().item() * 100

        # Text-to-image
        t2i_topk = sims.T.topk(k, dim=1).indices
        t2i_correct = (t2i_topk == gt.unsqueeze(1)).any(dim=1)
        t2i_recall = t2i_correct.float().mean().item() * 100

        print(f"  R@{k}: I→T {i2t_recall:.1f}%, T→I {t2i_recall:.1f}%")

    print("\n" + "=" * 50)
    print("Quick test completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
