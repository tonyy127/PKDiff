# PKDiff-for-multimodal-Remote-Sensing-Segmentation
The code is currently being organized and supplemented.


```markdown
# 多模态遥感分割（DDP）训练

基于 PyTorch DistributedDataParallel (DDP) 的多卡训练与评测脚本，支持单机多卡/多机多卡、Linear Warmup → Cosine 学习率调度、按参数名分组学习率、分布式评测聚合以及 NVML 显卡信息打印。


```
repo_root/
├── train.py                       # 训练入口
├── networks/network.py            # NetworkConfig, Network
├── trainer.py                     # Trainer, TrainerConfig
├── utils1.py                      # 数据路径/常量（train_ids/test_ids/BATCH_SIZE/...）
├── utils/evaluation.py            # Evaluator, denoise, metrics, accuracy  
├── eval_patch_dataset.py          # EvalPatchDataset
└── requirements.txt               # 依赖文件
```

## 2. 创建环境

```bash
# ① 新建 & 激活环境
conda create -n pkdiff python=3.10 -y
conda activate pkdiff

# ② 先装重依赖（带好二进制）
conda install -c conda-forge rasterio affine -y
```

## 3. 安装 requirements.txt 依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

## 4. 开始训练

### 单机多卡（以 4 卡为例）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 train.py
```

