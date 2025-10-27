# PKDiff-for-multimodal-Remote-Sensing-Segmentation
The code is currently being organized and supplemented.


🛰️ 多模态遥感分割（DDP）训练

基于 PyTorch DistributedDataParallel (DDP) 的多卡训练与评测脚本：支持 单机多卡 / 多机多卡、Linear Warmup → Cosine 学习率调度、按参数名分组 LR、分布式评测聚合与 NVML 显卡信息打印。能跑，多卡快。

1) 项目简介（分布式训练）

采用 torch.distributed 的 NCCL 后端；torchrun 启动，每进程绑一张 GPU（读 RANK/WORLD_SIZE/LOCAL_RANK）。

训练阶段：按步学习率（Linear warmup → Cosine）；评测阶段：各 rank 预测向量 all_gather 到 rank0 统一计算指标并广播。

代码结构（建议）：

repo_root/
├─ train.py
├─ networks/network.py            # NetworkConfig, Network
├─ trainer.py                     # Trainer, TrainerConfig
├─ utils1.py                      # 数据路径/常量（train_ids/test_ids/BATCH_SIZE/...）
├─ utils/evaluation.py            # Evaluator, denoise, metrics, accuracy
├─ eval_patch_dataset.py          # EvalPatchDataset
└─ requirements.txt

2) 创建环境（推荐：conda + pip）

rasterio/GDAL 比较“重”，先用 conda-forge 装它们，剩下的走 pip。

# ① 新建 & 激活环境
conda create -n rsddp python=3.10 -y
conda activate rsddp

# ② 先装重依赖（带好二进制）
conda install -c conda-forge rasterio affine -y

3) 安装 requirements.txt 依赖

项目根目录执行：

pip install -r requirements.txt


说明：requirements.txt 顶部包含
--extra-index-url https://download.pytorch.org/whl/cu121
对应 CUDA 12.1 的 PyTorch 轮子。如果你是 CPU-only 或其它 CUDA 版本，请删除/修改该行并相应调整 torch/torchvision 版本标记（如去掉 +cu121）。

4) 开始训练
单机多卡（以 4 卡为例）
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 train.py

多机多卡（示例：2 机 × 每机 4 卡）

在 node0（master）：

MASTER_ADDR=node0.host
MASTER_PORT=29500

torchrun --nnodes=2 --node_rank=0 \
         --nproc_per_node=4 \
         --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
         train.py


在 node1：

MASTER_ADDR=node0.host
MASTER_PORT=29500

torchrun --nnodes=2 --node_rank=1 \
         --nproc_per_node=4 \
         --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
         train.py


没有 InfiniBand 时可加：export NCCL_IB_DISABLE=1（更省事）。
