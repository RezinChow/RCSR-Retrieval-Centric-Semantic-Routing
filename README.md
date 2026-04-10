## Overview

This project implements a federated cross-modal retrieval framework with the following key components:
- **Router Module**: Semantic routing for cross-modal alignment
- **Minimax Fairness**: Robust-fair game for handling Non-IID data
- **Lightweight Personalization**: Client-specific adaptation

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## Quick Start

### 1. Environment Check

```bash
python check_environment.py
```

### 2. Data Preparation

**Flickr30K Dataset:**

Download the dataset and place it in `./datasets/flickr30k/`:

```bash
mkdir -p datasets/flickr30k

# Download Karpathy split annotations
wget https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
unzip caption_datasets.zip -d datasets/

# Download images from official source:
# https://shannon.cs.illinois.edu/DenotationGraph/
```

Expected directory structure:
```
datasets/flickr30k/
├── flickr30k_images/           # Image folder
└── dataset_flickr30k.json      # Karpathy split annotations
```

### 3. Running Experiments

**Quick Test (10 clients, S4 setting):**
```bash
python train.py --config configs/flickr30k/main/rcsr_full_s4.yaml
```

**Baseline Comparison (FedAvg):**
```bash
python train.py --config configs/flickr30k/baselines/fedavg_s4.yaml
```

**Ablation Study (w/o Router):**
```bash
python train.py --config configs/flickr30k/ablations/wo_router_s4.yaml
```

**Other Baseline Methods:**
```bash
# FedProx
python train.py --config configs/fedprox/flickr30k_s4.yaml

# FedPer
python train.py --config configs/fedper/flickr30k_s4.yaml

# MOON
python train.py --config configs/moon/flickr30k_s4.yaml

# CreamFL
python train.py --config configs/creamfl/flickr30k_s4.yaml

# FedMEKT
python train.py --config configs/fedmekt/flickr30k_s4.yaml

# pFedMe
python train.py --config configs/pfedme/flickr30k_s4.yaml

# MFCPL
python train.py --config configs/mfcpl/flickr30k_s4.yaml
```

**Batch Experiments:**
```bash
bash run_experiments.sh
```

### 4. Evaluation

```bash
python eval.py \
    --checkpoint outputs/rcsr_full_flickr30k_s4/best_model.pt \
    --config configs/flickr30k/main/rcsr_full_s4.yaml
```

## Experimental Settings

### S4 Setting (Extreme Non-IID + Missing Modality)
- **Non-IID**: Dirichlet distribution with α=0.1
- **Missing Modality**: 50% of clients lack one modality
- **Participation**: 50% of clients per round
- **Rounds**: 100 federated rounds (default)

### Supported Methods

| Method | Type | Reference |
|--------|------|-----------|
| **FedAvg** | Baseline | McMahan et al., 2017 |
| **FedProx** | Baseline | Li et al., 2020 |
| **FedPer** | Baseline | Arivazhagan et al., 2019 |
| **MOON** | Baseline | Li et al., 2021 |
| **CreamFL** | Comparison | Yu et al., 2023 |
| **FedMEKT** | Comparison | Zhang et al., 2023 |
| **pFedMe** | Comparison | Fallah et al., 2020 |
| **pFedMMA** | Comparison | - |
| **MFCPL** | Comparison | Information Fusion, 2025 |

## Repository Structure

```
.
├── train.py              # Main training script
├── eval.py               # Evaluation script
├── evaluation.py         # Evaluation utilities
├── check_environment.py  # Environment verification
├── requirements.txt      # Dependencies
├── run_experiments.sh    # Batch experiment script
├── configs/              # Configuration files
│   ├── flickr30k/       # Flickr30K experiments
│   ├── mscoco/          # MS COCO experiments
│   ├── msrvtt/          # MSR-VTT experiments
│   ├── creamfl/         # CreamFL baseline
│   ├── fedmekt/         # FedMEKT baseline
│   ├── mfcpl/           # MFCPL baseline
│   ├── pfedme/          # pFedMe baseline
│   ├── fedper/          # FedPer baseline
│   ├── fedprox/         # FedProx baseline
│   └── moon/            # MOON baseline
├── data/                 # Data loading and partitioning
├── models/               # Model definitions (CLIP + Adapters + Router)
├── fed/                  # Federated learning components
│   ├── client.py         # FL client
│   ├── server.py         # FL server with RCSR
│   ├── creamfl_compare/  # CreamFL implementation
│   ├── fedmekt_compare/  # FedMEKT implementation
│   ├── mfcpl/            # MFCPL implementation
│   ├── pfedme_compare/   # pFedMe implementation
│   ├── pfedmma_compare/  # pFedMMA implementation
│   ├── fedper/           # FedPer implementation
│   ├── fedprox/          # FedProx implementation
│   └── moon/             # MOON implementation
└── utils/                # Utilities (metrics, logging)
```

## Notes

- The first run will download CLIP model weights automatically
- wandb logging is disabled by default; enable in config files if needed
- Partition files will be generated automatically if not present
- Adjust `num_clients`, `num_rounds`, and other hyperparameters in config files as needed

## File List

For a complete list of files, see [FILELIST.md](FILELIST.md).
