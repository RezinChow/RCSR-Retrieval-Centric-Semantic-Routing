"""
Evaluation script for trained models
"""

import os
import argparse
import json
import torch
import clip
from typing import Dict

from data import get_dataset
from models import create_clip_model
from utils import load_config, RetrievalEvaluator, compute_retrieval_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Override data root path"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset to evaluate on (for transfer evaluation)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override config
    if args.data_root is not None:
        config['data']['data_root'] = args.data_root
    if args.dataset is not None:
        config['data']['dataset'] = args.dataset

    # Device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load CLIP preprocessing
    _, preprocess = clip.load(config['model']['backbone'], device=device)

    # Load dataset
    print(f"Loading {config['data']['dataset']} {args.split} split...")
    dataset = get_dataset(
        dataset_name=config['data']['dataset'],
        data_root=config['data']['data_root'],
        split=args.split,
        transform=preprocess,
        max_text_length=config['data'].get('max_text_length', 77),
    )
    print(f"Loaded {len(dataset)} samples")

    # Create model
    print("Creating model...")
    model = create_clip_model(
        backbone=config['model']['backbone'],
        embed_dim=config['model']['embed_dim'],
        adapter_dim=config['model']['adapter_dim'],
        freeze_backbone=config['model']['freeze_backbone'],
        use_adapters=config['model']['use_adapters'],
        device=device,
    )

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if 'model_state' in checkpoint:
        model.load_trainable_state_dict(checkpoint['model_state'])
    else:
        # Try loading directly
        model.load_trainable_state_dict(checkpoint)

    print("Checkpoint loaded")

    # Create evaluator
    evaluator = RetrievalEvaluator(
        model=model,
        test_dataset=dataset,
        batch_size=args.batch_size,
        device=device,
    )

    # Evaluate
    print("Evaluating...")
    metrics = evaluator.evaluate(k_values=[1, 5, 10])

    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Dataset: {config['data']['dataset']}")
    print(f"Split: {args.split}")
    print(f"Samples: {len(dataset)}")
    print("-" * 50)

    print("\nImage-to-Text Retrieval:")
    print(f"  R@1:  {metrics['i2t_r@1']:.2f}%")
    print(f"  R@5:  {metrics['i2t_r@5']:.2f}%")
    print(f"  R@10: {metrics['i2t_r@10']:.2f}%")

    print("\nText-to-Image Retrieval:")
    print(f"  R@1:  {metrics['t2i_r@1']:.2f}%")
    print(f"  R@5:  {metrics['t2i_r@5']:.2f}%")
    print(f"  R@10: {metrics['t2i_r@10']:.2f}%")

    print("\nMean Retrieval:")
    print(f"  R@1:  {metrics['mean_r@1']:.2f}%")
    print(f"  R@5:  {metrics['mean_r@5']:.2f}%")
    print(f"  R@10: {metrics['mean_r@10']:.2f}%")

    print("=" * 50)

    # Save results
    if args.output is not None:
        results = {
            'checkpoint': args.checkpoint,
            'dataset': config['data']['dataset'],
            'split': args.split,
            'num_samples': len(dataset),
            'metrics': metrics,
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
