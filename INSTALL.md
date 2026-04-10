
## Step-by-Step Installation

### 1. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n rcsr python=3.10
conda activate rcsr

# Or using venv
python -m venv venv
source venv/bin/activate
```

### 2. Install PyTorch

```bash
# With CUDA 11.8
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118

# Or for CPU only
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install CLIP

```bash
pip install git+https://github.com/openai/CLIP.git
```

### 4. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python -c "import torch; import clip; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch: 2.0.0+cu118
CUDA available: True
```

## Dataset Setup

### Flickr30K

1. Download images from [official source](https://shannon.cs.illinois.edu/DenotationGraph/)
2. Download Karpathy split annotations:
   ```bash
   wget https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
   unzip caption_datasets.zip -d datasets/
   ```
3. Organize as:
   ```
   datasets/flickr30k/
   ├── flickr30k_images/
   └── dataset_flickr30k.json
   ```

### MS COCO (Optional)

1. Download from [COCO website](https://cocodataset.org/)
2. Place in `datasets/coco/`

## Quick Test

Run a quick test to verify everything works:

```bash
python test_train.py
```

This will run a minimal training loop without requiring full dataset.

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config files
- Use smaller CLIP model: change `backbone: "ViT-B/32"` to `"RN50"`

### CLIP Download Issues
- If behind firewall, manually download CLIP weights from OpenAI

### Data Loading Errors
- Ensure dataset paths in config files match your setup
- Check JSON annotation file format matches Karpathy split
