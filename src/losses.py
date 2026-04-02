import torch
import torch.nn.functional as F

def compute_rdm_reg(z, mu_target=-1.0, sigma=1.0, num_projections=1024):
    """
    计算 Rectified Distribution Matching Regularization (RDMReg)
    利用 Sliced Wasserstein 距离，强制特征在经过 ReLU 后匹配目标截断广义高斯分布。
    """
    # 1. 展平 Batch 和 Sequence 维度：[B, N, D] -> [B*N, D]
    z_flat = z.view(-1, z.size(-1))
    D = z_flat.size(1)
    
    # 2. 隐空间的 ReLU 激活：强制特征具备非负稀疏性
    z_rect = F.relu(z_flat)
    
    # 3. 动态采样理想的 RGG (Rectified Generalized Gaussian) 目标分布
    # 在相同的设备上生成形状一致的噪声，加上目标均值漂移，再做 ReLU
    y_unrect = torch.randn_like(z_rect) * sigma + mu_target
    y_rect = F.relu(y_unrect)
    
    # 4. Sliced Wasserstein 投影匹配 (降维打击)
    # 随机生成 num_projections 个 1 维方向，并归一化
    directions = torch.randn(D, num_projections, device=z.device)
    directions = F.normalize(directions, p=2, dim=0)
    
    # 投影：高维分布 -> 多个1维分布  [B*N, num_projections]
    z_proj = torch.matmul(z_rect, directions)
    y_proj = torch.matmul(y_rect, directions)
    
    # 对每个1维投影进行独立排序
    z_sorted, _ = torch.sort(z_proj, dim=0)
    y_sorted, _ = torch.sort(y_proj, dim=0)
    
    # 计算均方误差 (1D Wasserstein-2 距离)
    rdm_loss = F.mse_loss(z_sorted, y_sorted)
    
    return rdm_loss