import torch
import torch.nn.functional as F
from utils.evaluation import Evaluator, segmentation_cross_entropy, noise_mse,segmentation_dice
from utils.utils import diffuse, get_patch_indices, dynamic_range

@torch.no_grad()
def _rand_choice_idx(n, k, device):
    if k >= n:
        return torch.arange(n, device=device)
    return torch.randperm(n, device=device)[:k]

def pkd_loss(P_pred, P_gt, sigma=1.0, max_samples=8192, unbiased=True):
    """
    P_pred, P_gt: [N, C] （预测概率/one-hot），在类别概率空间 R^C 比较分布
    sigma: 标量核宽度（先用 1.0，后续可做 median heuristic）
    max_samples: 子采样上限，控制算量/显存
    unbiased: U-统计无偏估计（去对角）
    返回：标量 IMMD 距离（可微）
    """
    assert P_pred.dim() == 2 and P_gt.dim() == 2 and P_pred.size(1) == P_gt.size(1)
    device = P_pred.device
    Np = P_pred.size(0)
    Ng = P_gt.size(0)

    idx_p = _rand_choice_idx(Np, max_samples, device)
    idx_g = _rand_choice_idx(Ng, max_samples, device)
    X = P_pred[idx_p]  # [np, C] 有梯度
    Y = P_gt[idx_g]    # [ng, C] 无梯度也可

    # pairwise squared distances
    D_xx = torch.cdist(X, X, p=2)**2  # [np, np]
    D_yy = torch.cdist(Y, Y, p=2)**2  # [ng, ng]
    D_xy = torch.cdist(X, Y, p=2)**2  # [np, ng]

    inv_s = 1.0 / (sigma * sigma)
    K_xx = (1.0 + D_xx * inv_s).rsqrt()
    K_yy = (1.0 + D_yy * inv_s).rsqrt()
    K_xy = (1.0 + D_xy * inv_s).rsqrt()

    if unbiased:
        np_ = K_xx.size(0)
        ng_ = K_yy.size(0)
        sum_xx = (K_xx.sum() - K_xx.diag().sum()) / (max(np_ * (np_ - 1), 1))
        sum_yy = (K_yy.sum() - K_yy.diag().sum()) / (max(ng_ * (ng_ - 1), 1))
    else:
        sum_xx = K_xx.mean()
        sum_yy = K_yy.mean()

    sum_xy = K_xy.mean()
    T_hat = sum_xx + sum_yy - 2.0 * sum_xy
    return T_hat

class TrainerConfig:
    """
    Config settings (hyperparameters) for training.
    """
    # optimization parameters
    # max_epochs = 100
    # batch_size = 4
    # learning_rate = 1e-5
    # momentum = None
    # weight_decay = 0.001 
    # grad_norm_clip = 0.95
    # 
    # # learning rate decay params
    # lr_decay = True
    # lr_decay_gamma = 0.98
    # 
    # # network
    # network = 'unet'

    # diffusion other settings
    # train_on_n_scales = None
    # not_recursive = False

    # checkpoint settings
    # checkpoint_dir = 'output/checkpoints/'
    # log_dir = 'output/logs/'
    # load_checkpoint = None
    # checkpoint = None
    # weights_only = False

    # data
    # dataset_selection = 'vaihingen'  #  uavid

    # other
    # eval_every = 2
    # save_every = 2
    # seed = 0
    # n_workers = 8

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, network_config, config, validation_data_loader=None):
        self.model = model
        self.network_config = network_config
        self.config = config
        #self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        #self.device = config.device


    def denoise_scales(self, model, network_config, images, seg_gt_one_hot, optimizer, scheduler=None):
        """Denoises all scales for a single timestep（替换版：稳态 & 可微IMMD）"""

        # 1) 计算多尺度尺寸（小到大）
        scale_sizes = [
            (
                images.shape[2] // (2 ** (network_config.n_scales - i - 1)),
                images.shape[3] // (2 ** (network_config.n_scales - i - 1))
            )
            for i in range(network_config.n_scales)
        ]

        # 2) 初始化上一次分割（随机），放到正确设备/精度
        seg_previous_scaled = torch.rand(
            images.shape[0], network_config.n_classes, images.shape[2], images.shape[3],
            device=images.device, dtype=images.dtype
        )

        last_losses = {}  # 返回一个代表性的 losses（最后一个patch为准）

        # 3) 递归去噪
        for timestep in range(network_config.n_timesteps):
            # 记录当前 step 的每个尺度的损失（可选）
            for scale in range(network_config.n_scales):
                # === 每个 scale 先 zero_grad，累计所有 patch 再 step ===
                optimizer.zero_grad()
                accum_loss = 0.0
                n_patches = 0

                # 3.1 缩放到当前尺度
                images_scaled = F.interpolate(images, size=scale_sizes[scale], mode='bilinear', align_corners=False)
                seg_gt_scaled = F.interpolate(seg_gt_one_hot, size=scale_sizes[scale], mode='bilinear', align_corners=False)
                seg_previous_scaled = F.interpolate(seg_previous_scaled, size=scale_sizes[scale], mode='bilinear', align_corners=False)

                patch_indices = get_patch_indices(scale_sizes[scale], network_config.max_patch_size, overlap=False)

                # 3.2 准备容器（CPU上累计，节省显存；结束再平均）
                seg_denoised = torch.zeros_like(seg_previous_scaled, device='cpu')
                n_denoised   = torch.zeros_like(seg_previous_scaled, device='cpu')

                # === Patch 循环：只 backward，不 step ===
                for x, y, patch_size in patch_indices:
                    img_patch = images_scaled[:, :, x:x+patch_size, y:y+patch_size].detach().to(images.device, non_blocking=True)
                    seg_gt_patch = seg_gt_scaled[:, :, x:x+patch_size, y:y+patch_size].detach().to(images.device, non_blocking=True)
                    seg_patch_previous = seg_previous_scaled[:, :, x:x+patch_size, y:y+patch_size].detach().to(images.device, non_blocking=True).softmax(dim=1)

                    # 3.3 时间编码
                    t_scalar = (network_config.n_timesteps - (timestep + scale / network_config.n_scales)) / network_config.n_timesteps
                    B = img_patch.size(0)
                    t_broadcast = torch.full((B, 1, 1, 1), t_scalar, device=img_patch.device, dtype=img_patch.dtype)
                    t_mlp = torch.full((B,), t_scalar, device=img_patch.device, dtype=img_patch.dtype)

                    # 3.4 扩散与噪声
                    seg_patch_diffused = diffuse(seg_patch_previous, t_broadcast).to(img_patch.device)
                    noise_gt = seg_patch_diffused - seg_gt_patch

                    # 3.5 前向：预测噪声 & 去噪后的分布
                    noise_predicted = model(seg_patch_diffused, img_patch, t_mlp)
                    seg_patch_denoised = seg_patch_diffused - noise_predicted  # 可能不在概率单纯形

                    # 1) 噪声MSE
                    loss_noise = noise_mse(noise_predicted, noise_gt)
                    
                    # 2) 分割：把去噪结果投回概率单纯形，再 NLLLoss（别把“伪logits”喂 CE）
                    p = (seg_patch_diffused - noise_predicted).clamp_min(1e-6)
                    p = p / p.sum(dim=1, keepdim=True)
                    logp = torch.log(p)
                    target_idx = seg_gt_patch.argmax(dim=1)
                    loss_ce = F.nll_loss(logp, target_idx)
                    
                    # 新增：3) Dice损失（用投影后的 p 和 target_idx）
                    loss_dice = segmentation_dice(p, target_idx)
                    
                    # 3) IMMD：对预测开梯度（不 detach），GT 可 detach；像素展平到 [N,C]
                    P_pred = p.permute(0, 2, 3, 1).reshape(-1, p.size(1))                 # 有梯度
                    P_gt   = seg_gt_patch.detach().permute(0, 2, 3, 1).reshape(-1, seg_gt_patch.size(1))
                    
                    lambda_immd = getattr(network_config, "lambda_immd", 0.075)
                    sigma_immd  = getattr(network_config, "sigma_immd", 1.0)
                    max_samples = getattr(network_config, "immd_max_samples", 8192)
                    lambda_dice = getattr(network_config, "lambda_dice", 1.0)  # 默认1.0，可调整为0.5~2.0
                    
                    # print("lambda_immd:  ",lambda_immd)
                    
                    loss_pkd = pkd_loss(P_pred, P_gt, sigma=sigma_immd, max_samples=max_samples, unbiased=True)
                    
                    # 4) 总损失：只 backward，不在 patch 内 step（等到该 scale 所有 patch 结束再 step）
                    #total_loss = loss_noise + loss_ce + lambda_immd * loss_pkd
                    total_loss = loss_noise + lambda_immd * loss_pkd
                    # total_loss = loss_noise
                    # total_loss = loss_noise + loss_ce
                    # total_loss = loss_noise + lambda_dice * loss_dice  # 当前代码的风格
                    total_loss.backward()


                    # 记录
                    accum_loss += float(total_loss.detach())
                    n_patches += 1
                    last_losses = {
                        'noise_mse': float(loss_noise.detach()),
                        'seg_cross_entropy': float(loss_ce.detach()),
                        'immd': float(loss_pkd.detach())
                    }

                    # 累计到CPU图（节省显存）
                    seg_patch_out_cpu = p.detach().cpu()
                    seg_denoised[:, :, x:x+patch_size, y:y+patch_size] += seg_patch_out_cpu
                    n_denoised[:, :, x:x+patch_size, y:y+patch_size] += 1

                # 3.7 该 scale 所有 patch 完成后：clip + step（只一次）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=getattr(network_config, "grad_clip", 1.0))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()   # ✅ 与真实的优化器 step 一一对应
                optimizer.zero_grad()

                # 3.8 平均并更新上一轮分割（回到CPU累计张量上做平均，再拷回GPU）
                seg_denoised = seg_denoised / (n_denoised + 1e-6)
                seg_previous_scaled = seg_denoised.to(images.device, non_blocking=True)

        return seg_previous_scaled, last_losses
