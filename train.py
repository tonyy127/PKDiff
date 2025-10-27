from tqdm import tqdm
import time
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from networks.network import NetworkConfig, Network
from utils1 import *
from torch.autograd import Variable
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener
import nvidia_smi
from trainer import Trainer, TrainerConfig
from utils.evaluation import Evaluator, denoise
from eval_patch_dataset import EvalPatchDataset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        rank = 0
        world_size = 1
        local_rank = 0
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

rank, world_size, local_rank = init_distributed_mode()
is_main_process = (rank == 0)

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(local_rank)
name = nvidia_smi.nvmlDeviceGetName(handle).decode()
print(f"Device[{local_rank}]: {name}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network_config = NetworkConfig(
    n_timesteps=25,
    n_scales=3,
    max_patch_size=128,
    scale_procedure='loop',
    n_classes=6,
    image_channels=4,
)
trainer_config = TrainerConfig(

)
net = Network(network_config).to(device)
net = DDP(net, device_ids=[local_rank], output_device=local_rank)
params = 0
for name, param in net.named_parameters():
    params += param.nelement()

if is_main_process:
    print("\n=== Network Configuration ===")
    for key, value in vars(network_config).items():
        print(f"{key}: {value}")
    print("params ：: ",params)
    # Load the datasets
    print("training : ", train_ids)
    print("testing : ", test_ids)
    print("BATCH_SIZE : ", BATCH_SIZE)
    print("TEST_STRIDE : ", TEST_STRIDE)
    
train_set = ISPRS_dataset(train_ids, cache=CACHE)
train_sampler = DistributedSampler(train_set)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, 
                          sampler=train_sampler, pin_memory=True, 
                          persistent_workers=True, prefetch_factor=2)

base_lr = 3e-4
weight_decay = 0.01

# 分组学习率：保持你原先“_D 用 base_lr，其他用半 lr”的逻辑
params = []
for name, p in net.named_parameters():
    if not p.requires_grad: 
        continue
    lr = base_lr if ('_D' in name) else (base_lr * 0.5)
    params.append({'params': [p], 'lr': lr})

optimizer = optim.AdamW(
    params,
    betas=(0.9, 0.999),
    weight_decay=weight_decay,
    eps=1e-8
)


EPOCHS = 100  # 你下面 train(..., 50, ...) 已经固定了
steps_per_epoch = len(train_loader)                      # 注意：这是 "迭代步/epoch"，非样本数
accum = 1                                               # 若你用梯度累积>1，请写实际值
total_updates = (steps_per_epoch * EPOCHS) // accum     # 总的“优化器更新步”数
warmup_updates = max(1, int(0.05 * total_updates))      # 5% warmup，可 3%~10% 微调

scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(optimizer, start_factor=0.1, total_iters=warmup_updates),
        CosineAnnealingLR(optimizer, T_max=(total_updates - warmup_updates))
    ],
    milestones=[warmup_updates]
)

def distributedifftest(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE, log_path="/root/autodl-tmp/logs/metrics.txt"):
    """
    分布式评测（仅 all=False）：各 rank 计算 patch 级预测，最终把 pred/gts 向量 gather 到 rank0，
    由 rank0 调用你提供的 metrics() 打印并返回 accuracy。其余 rank 返回同一个 accuracy 数值。
    """
    assert all is False, "当前版本仅支持 all=False（训练期快速评测）。"

    def is_dist():
        return dist.is_available() and dist.is_initialized()
    def is_main():
        return (not is_dist()) or (dist.get_rank() == 0)

    # Dataset / Sampler / DataLoader
    ds = EvalPatchDataset(
        ids=test_ids,
        window_size=window_size,
        stride=stride,
        dataset_name=DATASET,
        data_folder_tmpl=DATA_FOLDER,
        dsm_folder_tmpl=DSM_FOLDER,
        eroded_folder_tmpl=ERODED_FOLDER,
        convert_from_color_fn=convert_from_color
    )
    sampler = DistributedSampler(ds, shuffle=False, drop_last=False) if is_dist() else None
    loader = DataLoader(
        ds, batch_size=max(1, batch_size), 
        sampler=sampler, 
        shuffle=(sampler is None),
        num_workers=4, pin_memory=True, 
        persistent_workers=True,          
        #prefetch_factor=2,   
    )

    model_eval = net.module if hasattr(net, "module") else net
    was_training = model_eval.training
    model_eval.eval()

    preds_chunks = []
    gts_chunks   = []

    with torch.inference_mode():
        for images_4ch, gt_patch in loader:
            images_4ch = images_4ch.to(device, non_blocking=True)  # [B,4,h,w]
            gt_patch   = gt_patch.to(device, non_blocking=True)    # [B,h,w]

            logits = denoise(model_eval, device, network_config, images_4ch)  # 你的前向
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            pred = logits.argmax(dim=1)  # [B,h,w]

            # 扁平化为 numpy，尽量少占内存（按批次追加）
            preds_chunks.append(pred.detach().cpu().numpy().ravel())
            gts_chunks.append(gt_patch.detach().cpu().numpy().ravel())

            del images_4ch, gt_patch, logits, pred

    # 本 rank 的拼接结果
    local_preds = np.concatenate(preds_chunks) if len(preds_chunks) else np.empty((0,), dtype=np.int64)
    local_gts   = np.concatenate(gts_chunks)   if len(gts_chunks)   else np.empty((0,), dtype=np.int64)

    # 收集到 rank0，用你的 metrics() 统一打印与计算
    if is_dist():
        world_size = dist.get_world_size()
        obj_local = (local_preds, local_gts)
        obj_list = [None for _ in range(world_size)]
        dist.all_gather_object(obj_list, obj_local)   # ✅ 各 rank 都这么写
    
        if is_main():
            all_preds = [p for p, _ in obj_list if p is not None]
            all_gts   = [g for _, g in obj_list if g is not None]
            preds_all = np.concatenate(all_preds) if len(all_preds) else np.empty((0,), dtype=np.int64)
            gts_all   = np.concatenate(all_gts)   if len(all_gts)   else np.empty((0,), dtype=np.int64)
            acc = metrics(preds_all, gts_all, log_path=log_path)
        else:
            acc = None
    
        # 广播给所有 rank
        obj = [acc]
        dist.broadcast_object_list(obj, src=0)
        acc = obj[0]
    else:
        acc = metrics(local_preds, local_gts, log_path=log_path)

    if was_training:
        model_eval.train()
    return acc

def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()

    iter_ = 0
    MIoU_best = 0.00

    trainer = Trainer(net, network_config, config=trainer_config)

    for e in range(1, epochs + 1):
        if isinstance(train_loader.sampler, torch.utils.data.DistributedSampler):
            train_loader.sampler.set_epoch(e)
        net.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {e}/{epochs}",
                    disable=not is_main_process)
        for batch_idx, (data, dsm, target) in pbar:
            data = data.cuda(non_blocking=True)
            dsm = dsm.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            seg_gt_one_hot = F.one_hot(target, num_classes=6 + 1).permute(0, 3, 1, 2)[:, :-1, :,
                             :].float()  # make one hot (if remove void class [:,:-1,:,:])
            images = torch.cat([data, dsm.unsqueeze(1)], dim=1)

            seg_denoised, loss_dict = trainer.denoise_scales(net, network_config, \
                                                                  images=images, seg_gt_one_hot=seg_gt_one_hot,
                                                                  optimizer=optimizer)
            scheduler.step() #########################################################################################
            output = seg_denoised

            noise_mse_val = loss_dict["noise_mse"]
            if torch.is_tensor(noise_mse_val):
                noise_mse_val = noise_mse_val.item()  # 直接转 float

            losses[iter_] = noise_mse_val
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_ + 1])

            if iter_ % 1 == 0:
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]

                if is_main_process:
                    print('\nTrain (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                        e, epochs, batch_idx, len(train_loader),
                        100. * batch_idx / len(train_loader), noise_mse_val, accuracy(pred, gt)))
            iter_ += 1

            del data, dsm, target, output, seg_gt_one_hot, loss_dict

        if e % 5 == 0:
            MIoU = distributedifftest(net, test_ids, all=False, stride=TEST_STRIDE, batch_size=BATCH_SIZE, 
                                      window_size=WINDOW_SIZE, log_path="/root/autodl-fs/mmd_loss/dice_metrics.txt")
            if is_main_process and MIoU is not None and MIoU > MIoU_best:
                model_eval = net.module if hasattr(net, "module") else net
                torch.save(model_eval.state_dict(), f'/root/autodl-fs/mmd_loss/1dice_{e}_iou_{MIoU:.4f}')
                MIoU_best = MIoU

    if is_main_process:
        print('MIoU_best: ', MIoU_best)


#####   train   ####
time_start = time.time()
train(net, optimizer, EPOCHS, scheduler)
time_end = time.time()
print('Total hours Cost: ', (time_end - time_start)/3600)

# ####   test   ####
# sd = torch.load('/root/autodl-tmp/modelsave/quzao_epoch40_miou_0.8019', map_location='cpu')
# missing, unexpected = net.module.load_state_dict(sd, strict=False)  # 或 strict=True
# print('missing:', missing, 'unexpected:', unexpected)
# net.eval()
# miou = distributedifftest(net, test_ids, all=False, stride=256, batch_size=BATCH_SIZE, 
#                           window_size=WINDOW_SIZE, log_path="/root/autodl-tmp/testlogs/metrics.txt")
# print("miou: ", miou)
