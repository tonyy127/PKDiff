# PKDiff-for-multimodal-Remote-Sensing-Segmentation
The code is currently being organized and supplemented.


```markdown
# 多模态遥感分割（DDP）训练

基于 PyTorch DistributedDataParallel (DDP) 的多卡训练与评测脚本，支持单机多卡/多机多卡、Linear Warmup → Cosine 学习率调度、按参数名分组学习率、分布式评测聚合以及 NVML 显卡信息打印。代码高效、稳定，适合多卡加速。

## 1. 项目简介（分布式训练）

- **技术栈**：采用 `torch.distributed` 的 NCCL 后端，通过 `torchrun` 启动，每进程绑定一张 GPU（通过 `RANK`、`WORLD_SIZE`、`LOCAL_RANK` 区分）。
- **训练阶段**：支持按步学习率调度（Linear Warmup → Cosine）。
- **评测阶段**：各 rank 的预测向量通过 `all_gather` 聚合到 rank0，统一计算指标并广播结果。
- **代码结构**（建议）：

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

## 2. 创建环境（推荐：conda + pip）

由于 `rasterio` 和 `GDAL` 依赖较重，建议先用 `conda-forge` 安装这些依赖，其余通过 `pip` 安装。

```bash
# ① 新建 & 激活环境
conda create -n rsddp python=3.10 -y
conda activate rsddp

# ② 先装重依赖（带好二进制）
conda install -c conda-forge rasterio affine -y
```

## 3. 安装 requirements.txt 依赖

在项目根目录执行：

```bash
pip install -r requirements.txt
```

**说明**：`requirements.txt` 顶部包含以下行，用于指定 CUDA 12.1 的 PyTorch 轮子：

```
--extra-index-url https://download.pytorch.org/whl/cu121
```

- **CPU-only 或其他 CUDA 版本用户**：请删除或修改该行，并根据需要调整 `torch` 和 `torchvision` 的版本标记（例如，去掉 `+cu121`）。

## 4. 开始训练

### 单机多卡（以 4 卡为例）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 train.py
```

### 多机多卡（示例：2 机 × 每机 4 卡）

#### 在 node0（master）：

```bash
MASTER_ADDR=node0.host
MASTER_PORT=29500

torchrun --nnodes=2 --node_rank=0 \
         --nproc_per_node=4 \
         --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
         train.py
```

#### 在 node1：

```bash
MASTER_ADDR=node0.host
MASTER_PORT=29500

torchrun --nnodes=2 --node_rank=1 \
         --nproc_per_node=4 \
         --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
         train.py
```

**提示**：如果没有 InfiniBand，可设置以下环境变量以禁用 NCCL 的 InfiniBand 支持，简化配置：

```bash
export NCCL_IB_DISABLE=1
```
```
