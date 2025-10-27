# PKDiff-for-multimodal-Remote-Sensing-Segmentation
The code is currently being organized and supplemented.


🛰️ 多模态遥感分割（DDP）训练

一句话：基于 PyTorch DistributedDataParallel (DDP) 的多卡训练与评测脚本，内置 warmup+Cosine 学习率调度、按参数名分组 LR、分布式评测聚合与 NVML 显卡信息打印。能跑，多卡快。

1) 项目简介（分布式训练）

使用 torch.distributed 的 NCCL 后端，支持 单机多卡 / 多机多卡。

每个进程绑定一张 GPU：通过 RANK / WORLD_SIZE / LOCAL_RANK 环境变量自动识别。

训练阶段可选 按步学习率调度（Linear warmup → Cosine），评测阶段 跨进程聚合指标。

2) 创建环境（推荐：conda + pip）

rasterio/GDAL 比较“重”，先用 conda-forge 装它们，剩下的用 pip。

# ① 新建 & 激活环境
conda create -n rsddp python=3.10 -y
conda activate rsddp

# ② 先装重依赖（带好二进制）
conda install -c conda-forge rasterio affine -y

3) 安装 requirements.txt 依赖

项目根目录下执行：

pip install -r requirements.txt


说明：requirements.txt 顶部已包含
--extra-index-url https://download.pytorch.org/whl/cu121
对应 CUDA 12.1 的 PyTorch 轮子。若你是 CPU-only 或其它 CUDA 版本，请根据实际情况调整这一行以及 torch/torchvision 版本标记。

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


小贴士：没有 InfiniBand 时可临时
export NCCL_IB_DISABLE=1，网络更省心。
