```markdown
# PKDiff-for-Multimodal-Remote-Sensing-Segmentation
The code is currently being organized and supplemented.

# Multimodal Remote Sensing Segmentation (DDP) Training
Based on PyTorch DistributedDataParallel (DDP), this project provides scripts for multi-GPU training and evaluation, supporting single-machine multi-GPU and multi-machine multi-GPU setups, Linear Warmup → Cosine learning rate scheduling, parameter-grouped learning rates, distributed evaluation aggregation, and NVML GPU information printing.

## Project Structure
```
repo_root/
├── train.py                       # Training entry point
├── networks/network.py            # NetworkConfig, Network
├── trainer.py                     # Trainer, TrainerConfig
├── utils1.py                      # Data paths/constants (train_ids/test_ids/BATCH_SIZE/...)
├── utils/evaluation.py            # Evaluator, denoise, metrics, accuracy
├── eval_patch_dataset.py          # EvalPatchDataset
└── requirements.txt               # Dependency file
```


## 1. Create Environment
```bash
# ① Create & activate environment
conda create -n pkdiff python=3.10 -y
conda activate pkdiff
# ② Install heavy dependencies (with binaries)
conda install -c conda-forge rasterio affine -y
```

## 2. Install requirements.txt Dependencies
Run the following command in the project root directory:
```bash
pip install -r requirements.txt
```

## 3. Start Training
### Single Machine Multi-GPU (e.g., 4 GPUs)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 train.py
```
```
