#!/bin/bash
# Script to reproduce main experiments

set -e

echo "=========================================="
echo "Running Anonymous Code Experiments"
echo "=========================================="

# Create output directory
mkdir -p outputs

# Main experiment: RCSR on Flickr30K S4
echo "[1/3] Running RCSR (Full Method) - Flickr30K S4..."
python train.py --config configs/flickr30k/main/rcsr_full_s4.yaml

# Baseline: FedAvg
echo "[2/3] Running FedAvg Baseline - Flickr30K S4..."
python train.py --config configs/flickr30k/baselines/fedavg_s4.yaml

# Ablation: w/o Router
echo "[3/3] Running Ablation w/o Router - Flickr30K S4..."
python train.py --config configs/flickr30k/ablations/wo_router_s4.yaml

echo "=========================================="
echo "All experiments completed!"
echo "Results saved in ./outputs/"
echo "=========================================="
