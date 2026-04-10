"""
Main training script for RCSR federated cross-modal retrieval
"""

import os
# Set environment variables before importing other libraries
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import argparse
from typing import Dict, List, Optional
import torch
import clip
from torch.utils.data import Subset
from tqdm import tqdm

# Set multiprocessing start method
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from data import get_dataset, create_federated_partition, FederatedDataset
from models import create_clip_model, RCSRRouter
from fed import FederatedClient, FederatedServer, FedProxClient, FedPerClient, MOONClient

# Optional comparison methods (may not be available in minimal package)
try:
    from fed.creamfl_compare import CreamFLClient
except ImportError:
    CreamFLClient = None
try:
    from fed.fedmekt_compare import FedMEKTClient
except ImportError:
    FedMEKTClient = None
try:
    from fed.pfedme_compare import pFedMeClient
except ImportError:
    pFedMeClient = None
try:
    from fed.pfedmma_compare import pFedMMAClient
except ImportError:
    pFedMMAClient = None
try:
    from fed.MFCPL.mfcpl_client import MFCPLClient
except ImportError:
    MFCPLClient = None
from utils import (
    set_seed, load_config, Logger, RetrievalEvaluator,
    compute_fairness_metrics, AverageMeter
)
import math


def get_lr_schedule(
    round_num: int,
    total_rounds: int,
    base_lr: float,
    schedule_type: str = "cosine",
    warmup_rounds: int = 10,
    min_lr_ratio: float = 0.01,
) -> float:
    """
    Compute learning rate for current round.

    Args:
        round_num: Current round number (1-indexed)
        total_rounds: Total number of rounds
        base_lr: Base learning rate
        schedule_type: "cosine", "linear", "step", or "constant"
        warmup_rounds: Number of warmup rounds
        min_lr_ratio: Minimum LR as ratio of base_lr

    Returns:
        Learning rate for this round
    """
    min_lr = base_lr * min_lr_ratio

    # Warmup phase
    if round_num <= warmup_rounds:
        return base_lr * round_num / warmup_rounds

    # After warmup
    progress = (round_num - warmup_rounds) / (total_rounds - warmup_rounds)

    if schedule_type == "cosine":
        # Cosine annealing
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    elif schedule_type == "linear":
        # Linear decay
        lr = base_lr - (base_lr - min_lr) * progress
    elif schedule_type == "step":
        # Step decay at 50% and 75%
        if progress < 0.5:
            lr = base_lr
        elif progress < 0.75:
            lr = base_lr * 0.1
        else:
            lr = base_lr * 0.01
    else:  # constant
        lr = base_lr

    return lr


def parse_args():
    parser = argparse.ArgumentParser(description="RCSR Federated Training")
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
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed"
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=None,
        help="Override number of rounds"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use"
    )
    return parser.parse_args()


def setup_data(config: Dict, device: str):
    """
    Setup datasets and federated partition.

    Returns:
        train_fed_dataset: FederatedDataset for training
        test_dataset: Test dataset for evaluation
    """
    data_config = config['data']

    # Get CLIP preprocessing
    _, preprocess = clip.load(config['model']['backbone'], device=device)

    # Load train dataset
    # Check if multi-frame sampling is needed for video datasets
    num_frames = data_config.get('num_frames', 1)
    train_dataset = get_dataset(
        dataset_name=data_config['dataset'],
        data_root=data_config['data_root'],
        split="train",
        transform=preprocess,
        max_text_length=data_config.get('max_text_length', 77),
        num_frames=num_frames,
    )

    # Apply data subsampling if fraction < 1.0
    data_fraction = data_config.get('data_fraction', 1.0)
    if data_fraction < 1.0:
        import random
        original_size = len(train_dataset)
        subset_size = int(original_size * data_fraction)
        # Random sampling instead of taking first N samples
        random.seed(config['seed'])
        indices = random.sample(range(original_size), subset_size)
        train_dataset = Subset(train_dataset, indices)
        print(f"Using {subset_size}/{original_size} samples ({data_fraction*100:.0f}%) [random sampled]")

    # Load test dataset
    test_dataset = get_dataset(
        dataset_name=data_config['dataset'],
        data_root=data_config['data_root'],
        split="test",
        transform=preprocess,
        max_text_length=data_config.get('max_text_length', 77),
        num_frames=num_frames,
    )

    # Create federated partition
    partition_config = config['partition']
    missing_config = config['missing_modality']

    # Check if partition file is specified
    partition_file = partition_config.get('partition_file', None)
    if partition_file is not None:
        print(f"Using pre-defined partition file: {partition_file}")

    client_indices, modality_masks = create_federated_partition(
        dataset=train_dataset,
        num_clients=config['federation']['num_clients'],
        alpha=partition_config['alpha'],
        num_clusters=partition_config.get('num_clusters', 50),
        missing_rate=missing_config['missing_rate'] if missing_config['enabled'] else 0.0,
        seed=config['seed'],
        partition_file=partition_file,
    )

    # Create federated dataset
    train_fed_dataset = FederatedDataset(
        base_dataset=train_dataset,
        client_indices=client_indices,
        modality_masks=modality_masks,
    )

    return train_fed_dataset, test_dataset


def train_round(
    round_num: int,
    server: FederatedServer,
    clients: Dict[int, FederatedClient],
    selected_clients: List[int],
    config: Dict,
) -> Dict[str, float]:
    """
    Execute one federated training round.

    Returns:
        Round metrics dict
    """
    # Get global model and prototypes
    global_state = server.broadcast_global_model()
    global_img_proto, global_txt_proto = server.broadcast_global_prototypes()

    # Client updates storage
    client_updates = {}
    client_stats = {}
    client_losses = {}

    # Local training on selected clients
    for client_id in selected_clients:
        client = clients[client_id]

        # Receive global model
        client.receive_global_model(global_state)
        client.receive_global_prototypes(global_img_proto, global_txt_proto)

        # Store initial state
        initial_state = client.model.get_trainable_state_dict()

        # Local training
        train_result = client.local_train()

        # Compute update
        update = client.compute_update(initial_state)

        # Get client statistics for router
        stats = client.get_client_stats(
            update=update,
            loss=train_result['loss'],
        )

        client_updates[client_id] = update
        client_stats[client_id] = stats
        client_losses[client_id] = train_result['loss']

    # Server aggregation
    agg_metrics = server.aggregate(
        client_updates=client_updates,
        client_stats=client_stats,
        client_losses=client_losses,
    )

    # Compute average training loss
    avg_train_loss = sum(client_losses.values()) / len(client_losses)

    return {
        'train_loss': avg_train_loss,
        **agg_metrics,
    }


def evaluate(
    server: FederatedServer,
    evaluator: RetrievalEvaluator,
    clients: Optional[Dict[int, FederatedClient]] = None,
) -> Dict[str, float]:
    """
    Evaluate global model.

    Returns:
        Evaluation metrics dict
    """
    # Get global model
    model = server.get_global_model()

    # Update evaluator model
    evaluator.model = model

    # Global evaluation
    metrics = evaluator.evaluate()

    # Per-client evaluation (if clients provided)
    if clients is not None:
        # Broadcast global model to clients
        global_state = server.broadcast_global_model()
        for client in clients.values():
            client.receive_global_model(global_state)

        # Evaluate on each client
        client_metrics = {}
        for client_id, client in clients.items():
            client_result = client.evaluate()
            client_metrics[client_id] = client_result

        # Compute fairness metrics
        fairness = compute_fairness_metrics(client_metrics)
        metrics.update({f'fairness_{k}': v for k, v in fairness.items()})

    return metrics


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    if args.data_root is not None:
        config['data']['data_root'] = args.data_root
    if args.output_dir is not None:
        config['logging']['save_dir'] = args.output_dir
    if args.seed is not None:
        config['seed'] = args.seed
    if args.num_rounds is not None:
        config['federation']['num_rounds'] = args.num_rounds
    if args.device is not None:
        config['device'] = args.device

    # Set seed
    set_seed(config['seed'])

    # Device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup logger
    logger = Logger(
        save_dir=config['logging']['save_dir'],
        exp_name=config['logging']['exp_name'],
        use_wandb=config['logging'].get('use_wandb', False),
        wandb_project=config['logging'].get('wandb_project'),
        config=config,
    )

    logger.log(f"Starting experiment: {config['logging']['exp_name']}")
    logger.log(f"Config: {config}")

    # Setup data
    logger.log("Setting up data...")
    train_fed_dataset, test_dataset = setup_data(config, device)
    logger.log(f"Train dataset: {len(train_fed_dataset.base_dataset)} samples, "
               f"{train_fed_dataset.num_clients} clients")
    logger.log(f"Test dataset: {len(test_dataset)} samples")

    # Create model
    logger.log("Creating model...")
    model = create_clip_model(
        backbone=config['model']['backbone'],
        embed_dim=config['model']['embed_dim'],
        adapter_dim=config['model']['adapter_dim'],
        freeze_backbone=config['model']['freeze_backbone'],
        use_adapters=config['model']['use_adapters'],
        device=device,
    )

    # Create server
    logger.log("Creating server...")
    server = FederatedServer(
        model=model,
        num_clients=config['federation']['num_clients'],
        config=config,
        device=device,
    )

    # Create clients
    logger.log("Creating clients...")
    clients = {}
    
    # Get personalization config to determine client type
    personalization_config = config.get('personalization', {})
    method = personalization_config.get('method', None)
    
    # Determine client class based on method
    if method == 'fedprox':
        client_class = FedProxClient
        logger.log(f"Using FedProxClient with mu={personalization_config.get('mu', 0.1)}")
    elif method == 'fedper':
        client_class = FedPerClient
        logger.log(f"Using FedPerClient with personalized_keys={personalization_config.get('personalized_keys', ['adapter', 'prompt'])}")
    elif method == 'moon':
        client_class = MOONClient
        logger.log(f"Using MOONClient with moon_mu={personalization_config.get('moon_mu', 1.0)}, "
                   f"moon_temperature={personalization_config.get('moon_temperature', 0.5)}")
    elif method == 'creamfl':
        if CreamFLClient is None:
            raise ImportError("CreamFLClient not available. Please install the full package.")
        client_class = CreamFLClient
        logger.log(f"Using CreamFLClient with contrast_inter={personalization_config.get('contrast_inter', True)}, "
                   f"contrast_intra={personalization_config.get('contrast_intra', True)}")
    elif method == 'fedmekt':
        if FedMEKTClient is None:
            raise ImportError("FedMEKTClient not available. Please install the full package.")
        client_class = FedMEKTClient
        logger.log(f"Using FedMEKTClient with kd_weight={personalization_config.get('kd_weight', 0.5)}, "
                   f"kd_temperature={personalization_config.get('kd_temperature', 3.0)}")
    elif method == 'pfedme':
        if pFedMeClient is None:
            raise ImportError("pFedMeClient not available. Please install the full package.")
        client_class = pFedMeClient
        logger.log(f"Using pFedMeClient with lambda_reg={personalization_config.get('lambda_reg', 0.1)}")
    elif method == 'pfedmma':
        if pFedMMAClient is None:
            raise ImportError("pFedMMAClient not available. Please install the full package.")
        client_class = pFedMMAClient
        logger.log(f"Using pFedMMAClient")
    elif method == 'mfcpl':
        if MFCPLClient is None:
            raise ImportError("MFCPLClient not available. Please install the full package.")
        client_class = MFCPLClient
        logger.log(f"Using MFCPLClient")
    else:
        client_class = FederatedClient
        logger.log(f"Using standard FederatedClient (method={method})")
    
    client_config = {
        'batch_size': config['training']['batch_size'],
        'local_epochs': config['federation']['local_epochs'],
        'lr': config['training']['lr'],
        'weight_decay': config['training']['weight_decay'],
        'temperature': config['training']['temperature'],
        'lambda_align': config['training']['lambda_align'],
        'lambda_anchor': config['training']['lambda_anchor'],
        'lambda_stab': config['training']['lambda_stab'],
        'lambda_cma': config['training'].get('lambda_cma', 0.0),
        'lambda_clip_anchor': config['training'].get('lambda_clip_anchor', 0.0),
        # FedProx support
        'mu': personalization_config.get('mu', 0.0),
        # CreamFL support
        'contrast_inter': personalization_config.get('contrast_inter', True),
        'contrast_intra': personalization_config.get('contrast_intra', True),
        'interintra_weight': personalization_config.get('interintra_weight', 0.5),
        'kd_weight': personalization_config.get('kd_weight', 0.3),
        'creamfl_temperature': personalization_config.get('creamfl_temperature', 0.07),
        'feature_ema_decay': personalization_config.get('feature_ema_decay', 0.9),
        # FedMEKT support
        'kd_temperature': personalization_config.get('kd_temperature', 3.0),
        'alpha': personalization_config.get('alpha', 0.5),
        'beta': personalization_config.get('beta', 0.3),
        'use_contrastive': personalization_config.get('use_contrastive', True),
    }

    for client_id in range(config['federation']['num_clients']):
        client_dataset = train_fed_dataset.get_client_dataset(client_id)
        modality_mask = train_fed_dataset.get_client_modality_mask(client_id)

        # Create client model (copy of global model)
        client_model = create_clip_model(
            backbone=config['model']['backbone'],
            embed_dim=config['model']['embed_dim'],
            adapter_dim=config['model']['adapter_dim'],
            freeze_backbone=config['model']['freeze_backbone'],
            use_adapters=config['model']['use_adapters'],
            device=device,
        )

        clients[client_id] = client_class(
            client_id=client_id,
            model=client_model,
            dataset=client_dataset,
            modality_mask=modality_mask,
            config=client_config,
            device=device,
        )

    # Create evaluator
    evaluator = RetrievalEvaluator(
        model=model,
        test_dataset=test_dataset,
        batch_size=config['training']['batch_size'],
        device=device,
    )

    # Training loop
    logger.log("Starting training...")
    num_rounds = config['federation']['num_rounds']
    eval_every = config['eval']['eval_every']
    best_metric = 0.0

    # Learning rate schedule config
    lr_schedule_config = config.get('lr_schedule', {})
    use_lr_schedule = lr_schedule_config.get('enabled', False)
    schedule_type = lr_schedule_config.get('type', 'cosine')
    warmup_rounds = lr_schedule_config.get('warmup_rounds', 10)
    min_lr_ratio = lr_schedule_config.get('min_lr_ratio', 0.01)
    base_lr = config['training']['lr']

    if use_lr_schedule:
        logger.log(f"Using {schedule_type} LR schedule with warmup={warmup_rounds}, min_lr_ratio={min_lr_ratio}")

    for round_num in tqdm(range(1, num_rounds + 1), desc="Rounds"):
        # Update learning rate if using schedule
        if use_lr_schedule:
            current_lr = get_lr_schedule(
                round_num=round_num,
                total_rounds=num_rounds,
                base_lr=base_lr,
                schedule_type=schedule_type,
                warmup_rounds=warmup_rounds,
                min_lr_ratio=min_lr_ratio,
            )
            # Update all clients' learning rate
            for client in clients.values():
                client.lr = current_lr
        else:
            current_lr = base_lr

        # Select clients
        selected_clients = server.select_clients(
            config['federation']['participation_rate']
        )

        # Train round
        train_metrics = train_round(
            round_num=round_num,
            server=server,
            clients=clients,
            selected_clients=selected_clients,
            config=config,
        )
        train_metrics['lr'] = current_lr  # Log current LR

        # Evaluation
        eval_metrics = None
        if round_num % eval_every == 0 or round_num == num_rounds:
            eval_metrics = evaluate(server, evaluator, clients)

            # Check for best model
            current_metric = eval_metrics['mean_r@1']
            if current_metric > best_metric:
                best_metric = current_metric
                logger.save_checkpoint(
                    {
                        'round': round_num,
                        'model_state': server.global_state,
                        'metrics': eval_metrics,
                    },
                    filename="checkpoint.pt",
                    is_best=True,
                )

        # Log metrics
        logger.log_round(
            round_num=round_num,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            server_metrics=train_metrics,  # Server metrics included in train_metrics
        )

    # Final evaluation
    logger.log("Final evaluation...")
    final_metrics = evaluate(server, evaluator, clients)
    logger.log(f"Final metrics: {final_metrics}")

    # Save final checkpoint
    logger.save_checkpoint(
        {
            'round': num_rounds,
            'model_state': server.global_state,
            'metrics': final_metrics,
        },
        filename="final_checkpoint.pt",
    )

    # Finish logging
    logger.finish()
    logger.log(f"Training complete. Best R@1: {best_metric:.2f}")


if __name__ == "__main__":
    main()
