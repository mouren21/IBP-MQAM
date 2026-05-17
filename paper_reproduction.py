import os
import math
import random
import copy
import gc  
from concurrent.futures import ThreadPoolExecutor
import numpy as np


try:
    import ser

    _SER_AVAILABLE = True
except ImportError:
    _SER_AVAILABLE = False
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional, Union
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 使用 torch.amp.autocast('cuda') 替代已废弃的 torch.cuda.amp.autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.models as models
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# 确保顶部有这个 import
from scipy.optimize import minimize_scalar


def _dppo_feature_visit_order(feat_dim: int) -> List[int]:
    """DPPO 每步访问特征的顺序：固定 0..feat_dim-1，与 encoder/decoder 维序一致（收发不重排）。"""
    return list(range(int(feat_dim)))


def _decoder_first_linear_weight(decoder: nn.Module) -> Optional[torch.Tensor]:
    """SemanticDecoder：net[0] 为 Linear，返回 W1 形状 [out_dim, in_dim]（与 320.py / 406.py 一致）。"""
    if decoder is None:
        return None
    try:
        first = decoder.net[0]
        if isinstance(first, nn.Linear):
            return first.weight.detach()
    except Exception:
        return None
    return None


def _nsv_diag_task_var(Zc: torch.Tensor, W1: torch.Tensor) -> np.ndarray:
    """320.py: task_diag = ||W 的第 j 列||²，var_diag = Zc 列方差；对角近似 h·λ。"""
    n_m1 = max(Zc.shape[0] - 1, 1)
    var_diag = (Zc.pow(2).sum(dim=0) / n_m1).cpu().numpy()
    task_diag = (W1.float().pow(2).sum(dim=0)).cpu().numpy()
    return np.maximum(var_diag * task_diag, 1e-20)


def _nsv_pca_h_lam_Q(Zc: torch.Tensor, W1: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    406.py：对 Sigma_x = Zc^T Zc/(n-1) 特征分解，lambda_k 为特征值（降序），
    h_k = ||W1 q_k||_2^2（解码器第一层在特征向量 q_k 上的灵敏度）。
    返回 h, lam, Q（Q 的列与 h、lam 同序，均为大特征值在前）。
    """
    n = Zc.shape[0]
    d = Zc.shape[1]
    zf = Zc.double()
    wf = W1.double()
    n_m1 = max(n - 1, 1)
    sigma = (zf.T @ zf) / n_m1
    lam_asc, q_asc = torch.linalg.eigh(sigma)
    lam_desc = lam_asc.flip(dims=(0,))
    q_desc = q_asc.flip(dims=(1,))
    lam_np = np.maximum(lam_desc.cpu().numpy(), 1e-20)
    q_np = q_desc.cpu().numpy()
    w1_tilde = wf @ q_desc
    h_np = w1_tilde.pow(2).sum(dim=0).cpu().numpy()
    h_np = np.maximum(h_np, 1e-20)
    return h_np, lam_np, q_np


def get_nsv_power_closed_form(
        h: np.ndarray,
        lam: np.ndarray,
        g_sq: np.ndarray,
        n0: float,
        p_tot: float,
) -> np.ndarray:
    """
    406.py get_nsv 闭式注水（代数 NSV 最优功率分配）：
        p_i = max( sqrt(h_i * lambda_i * n0 / (nu * g_i^2)) - n0/g_i^2 , 0 )
    其中 nu* 由二分法使 sum_i p_i = P_tot。
    """
    h = np.maximum(np.asarray(h, dtype=np.float64), 1e-30)
    lam = np.maximum(np.asarray(lam, dtype=np.float64), 1e-30)
    g_sq = np.maximum(np.asarray(g_sq, dtype=np.float64), 1e-30)
    n0 = float(n0)
    p_tot = float(max(p_tot, 1e-6))

    def total_power(nu: float) -> float:
        nu = max(float(nu), 1e-200)
        inner = h * lam * n0 / (nu * g_sq + 1e-150)
        inner = np.maximum(inner, 0.0)
        terms = np.sqrt(inner) - n0 / g_sq
        return float(np.sum(np.maximum(terms, 0.0)))

    low, high = 1.0, 1.0
    guard = 0
    while total_power(high) > p_tot and guard < 220:
        high *= 10.0
        guard += 1
    guard = 0
    while total_power(low) < p_tot and guard < 220:
        low /= 10.0
        if low < 1e-150:
            break
        guard += 1
    for _ in range(80):
        mid = (low + high) / 2.0
        if total_power(mid) > p_tot:
            low = mid
        else:
            high = mid
    nu_opt = (low + high) / 2.0
    inner = h * lam * n0 / (nu_opt * g_sq + 1e-150)
    inner = np.maximum(inner, 0.0)
    return np.maximum(np.sqrt(inner) - n0 / g_sq, 0.0)


def nsv_mode_power_to_coordinate_weights(Q: np.ndarray, p_mode: np.ndarray) -> np.ndarray:
    """将各主模上的注水功率 p_k 映回原始坐标 j：w_j = sum_k Q[j,k]^2 * p_k（与特征索引对齐）。"""
    p_mode = np.asarray(p_mode, dtype=np.float64).ravel()
    q = np.asarray(Q, dtype=np.float64)
    k = min(q.shape[1], p_mode.shape[0])
    return np.sum((q[:, :k] ** 2) * p_mode[:k].reshape(1, -1), axis=1)


def compute_optimal_alpha_analytical(omega_batch: torch.Tensor, bit_allocation_batch: list,
                                     current_snr: float,
                                     feats_batch: Optional[torch.Tensor] = None,
                                     decoder: Optional[nn.Module] = None,
                                     use_nsv_weights: bool = True,
                                     nsv_bit_coupling: float = 0.2,
                                     nsv_weight_scheme: str = "406") -> float:
    """
    根据已确定的比特分配方案，连续、精确地求解当前 Batch 最优的 alpha。

    权重模式
    --------
    - **nsv_weight_scheme == "406"**（默认）：与 406/320 一致
        * 用当前 batch 特征 Z 中心化得 Sigma_x，特征分解得 lambda_k、正交基 Q；
        * h_k = ||W1 q_k||^2，W1 为 SemanticDecoder 第一层 Linear 权重；
        * 按 406 闭式 NSV 注水在模式上得功率 p_k（总预算 P_tot 取本 batch 已分配总比特，信道增益取 g^2=1）；
        * 将 p 映回原始维度 w_j = sum_k Q[j,k]^2 p_k，再乘 (1 + gamma * total_bits) 作为该分配项的 alpha 优化权重。
    - **nsv_weight_scheme == "diag"**：对角近似 w_j ∝ (diag(W^T W))_j * 列方差 * (1+gamma*bits)，不再做 PCA+注水。
    - **use_nsv_weights 为 False**：仅用 omega 加权（最简旧版）。
    """
    # 如果没有导入误码率计算模块，返回默认值
    if not _SER_AVAILABLE:
        return 0.5

    B = omega_batch.shape[0]
    suffix_bits = 8  # 对应论文中的 6-bit IBP: 2^(6/2)=8
    om = omega_batch.float()

    lam_vec: Optional[torch.Tensor] = None
    w_coord_np: Optional[np.ndarray] = None

    scheme = (nsv_weight_scheme or "406").lower()
    if scheme == "406" and decoder is None:
        scheme = "diag"

    if feats_batch is not None and use_nsv_weights:
        fb = feats_batch.float()
        if fb.shape[0] != B:
            fb = fb[:B]
        W1 = _decoder_first_linear_weight(decoder)
        if scheme == "406" and W1 is not None and fb.shape[0] >= 2:
            Zc = fb - fb.mean(dim=0, keepdim=True)
            try:
                Zc64 = Zc.to(dtype=torch.float64)
                W164 = W1.to(device=Zc64.device, dtype=torch.float64)
                h_np, lam_np, q_np = _nsv_pca_h_lam_Q(Zc64, W164)
                g_sq = np.ones(h_np.shape[0], dtype=np.float64)
                n0 = 1.0
                p_tot = 0.0
                for b in range(B):
                    for alloc in bit_allocation_batch[b]:
                        if len(alloc) >= 4 and alloc[1] > 0:
                            p_tot += float(alloc[1])
                p_tot = max(p_tot, 100.0)
                p_mode = get_nsv_power_closed_form(h_np, lam_np, g_sq, n0, p_tot)
                w_coord_np = nsv_mode_power_to_coordinate_weights(q_np, p_mode)
                w_coord_np = np.maximum(w_coord_np, 1e-20)
            except Exception:
                w_coord_np = None
        if w_coord_np is None and W1 is not None:
            Zc = fb - fb.mean(dim=0, keepdim=True)
            w_coord_np = _nsv_diag_task_var(Zc, W1.to(device=Zc.device))
        if w_coord_np is None:
            if fb.shape[0] > 1:
                lam_vec = fb.var(dim=0, unbiased=False).clamp(min=1e-8)
            else:
                lam_vec = fb.squeeze(0).abs().clamp(min=1e-8)

    # 1. 预提取：把所有真正分配了比特的特征提出来，避免优化器在内部循环中做无用功
    valid_allocs: List[Tuple[float, int, int]] = []
    total_w = 0.0
    for b in range(B):
        for alloc in bit_allocation_batch[b]:
            # alloc 的格式是 (feat_idx, total_bits, imp_bits, nsk_bits)
            if len(alloc) >= 4 and alloc[1] > 0:
                feat_idx, total_bits, imp_bits, nsk_bits = alloc[0], alloc[1], alloc[2], alloc[3]
                if w_coord_np is not None:
                    wc = float(w_coord_np[int(feat_idx)])
                    w = wc * (1.0 + float(nsv_bit_coupling) * float(total_bits))
                elif lam_vec is not None:
                    lam = float(lam_vec[feat_idx].item())
                    h = float(om[b, feat_idx].item())
                    w = h * lam * (1.0 + float(nsv_bit_coupling) * float(total_bits))
                else:
                    w = float(om[b, feat_idx].item())
                valid_allocs.append((w, imp_bits, nsk_bits))
                total_w += w

    # 如果没有任何特征被分配比特，直接返回默认 alpha
    if total_w == 0.0 or not valid_allocs:
        return 0.5

    # 2. 定义目标函数：针对给定 alpha 计算当前分配的加权总 MSE
    def objective(alpha: float) -> float:
        try:
            # 调用底层误码率公式
            _, p_prefix, p_suffix = ser.compute_theoretical_nsk_ser(current_snr, alpha, suffix_bits)
        except Exception:
            p_prefix, p_suffix = 0.05, 0.10

        p_prefix = max(0.0, min(1.0, p_prefix))
        p_suffix = max(0.0, min(1.0, p_suffix))

        total_loss = 0.0
        for w, imp_bits, nsk_bits in valid_allocs:
            # 结合重要比特和非重要比特的误码率
            Pi = 1.0 - (1.0 - p_prefix) ** (1.0 / max(1, imp_bits))
            Pu = 1.0 - (1.0 - p_suffix) ** (1.0 / max(1, nsk_bits))

            # 使用原代码中的理论 MSE 公式
            mse_i = _paper_mse_per_action(imp_bits, nsk_bits, Pi, Pu)
            total_loss += w * mse_i

        return total_loss / total_w

    # 3. 使用标量有界优化器，在 [0.1, 1.16] 范围内寻找使 objective 最小的 alpha
    res = minimize_scalar(objective, bounds=(0.1, 1.16), method='bounded')

    if res.success:
        return float(res.x)
    else:
        return 0.5


class DatasetWithIndex(Dataset):
    """包装 Dataset，__getitem__ 返回 (sample, label, index)，用于按索引取预计算的每图 omega。"""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        if isinstance(item, (list, tuple)):
            return (*item, idx)
        return item, idx


# IBP动作映射：固定 alpha=0.4、按欧氏距离排序，与 QAM 一致在 4dB 训练、全 SNR 评估
def build_ibp_action_space(max_bits):
    """
    构建IBP动作空间：(总比特, 重要比特)组合，0 <= total <= max_bits，含0比特(0,0)。
    排序：按 (total, imp) 相对 (0,0) 的欧氏距离升序排列（与 QAM 一致）。
    """
    actions = [(0, 0)]  # 0比特为合法动作
    for total in range(1, max_bits + 1):
        for imp in range(total + 1):
            actions.append((total, imp))
    # 欧氏距离 sqrt(total^2 + imp^2)，升序；tie-break：total 小优先，其次 imp 小优先
    actions.sort(key=lambda x: (x[0] ** 2 + x[1] ** 2, x[0], x[1]))
    index_to_action = {i: act for i, act in enumerate(actions)}
    action_to_index = {act: i for i, act in enumerate(actions)}
    return index_to_action, action_to_index


def build_ibp_action_space_for_snr(snr_db, max_bits, mse_data=None, optimal_alpha_data=None):
    """
    动作空间：按 (total, imp) 相对 (0,0) 的欧氏距离升序排列（与 QAM 一致，统一训练/评估）。
    固定 alpha=0.4，不再使用 MSE 表或最优 alpha。mse_data/optimal_alpha_data 保留兼容。
    """
    return build_ibp_action_space(max_bits)


# 双GPU时仅用5080：运行前设置 CUDA_VISIBLE_DEVICES=1，则本脚本只用 GPU1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True  # 固定输入尺寸时加速卷积

# ==================== Codebook-based quantization & dequantization (for QAM & IBP) ====================
"""
量化与码本（QAM vs IBP）：
- QAM：编码端 DitherQuantizer.quantize_feature 始终为均匀抖动量化，不使用码本。解码端 dequantize_from_int
  若存在 ibp_qam_codebooks.pth 则查表，否则线性反量化。故仅解码用码本时与编码端电平可能不一致。
- IBP：编码端 IBPQuantizer.quantize_with_ibp 在有码本时为无抖动、找最近码本电平；无码本时为无抖动均匀量化。
  解码端 dequantize_from_int 同上（有码本查表，否则线性）。编码与解码均可用码本，保持一致。
- 码本文件结构：{bit_width: 1D Tensor[2^bit_width]}。建议用 build_improved_codebooks.py 生成。

训练与评估：量化和调制逻辑一致，无差别。唯一差别为 IBP 评估可按 SNR 加载不同模型与动作空间，QAM 用同一模型。

语义重要性 omega 与失真尺度（本质，按论文来）：
- 论文 Fig.11 明确："the importance values of all the semantics in the image are normalized to have a **sum of 1**"。因此 Eq.(10) 中的 d = Σ ωi|ai−a'i|² 在 Σωi=1 下是**加权平均** MSE，尺度为 0～几，不是几十上百。
- 论文 Fig.6 的 "MSE is less than 1" 指**信道估计 MSE**（channel estimation），与语义失真 d 无关。
- **正确设定（论文一致）**：omega 做 min-max 后**再归一化为 sum=1**，L0=10（Table I）。这样 d 为加权平均、训练目标与论文一致，才能训出应有结果。

诊断「高 SNR 下 IBP 任务准确率仅 ~62% 而非 ~96%」：
- 若训练未用码本而评估时加载了码本，解码重建与训练分布不一致。可设 FORCE_LINEAR_DEQUANT_IBP=1
  使编码/解码均退化为均匀+线性；若准确率恢复则问题在码本与训练不匹配。
"""
try:
    _loaded_codebooks = torch.load("ibp_qam_codebooks.pth", map_location="cpu")
    # 统一成 {int(bit_width): 1D tensor}，在第一次使用时搬到 device 上
    CODEBOOKS: Dict[int, torch.Tensor] = {
        int(k): torch.as_tensor(v, dtype=torch.float32) for k, v in _loaded_codebooks.items()
    }
    del _loaded_codebooks
except Exception:
    CODEBOOKS = None


def dequantize_from_int(int_val: int, total_bits: int, device_: Optional[torch.device] = None) -> torch.Tensor:
    """
    给定整数 int_val 和比特数 total_bits，返回对应的重建连续值：
    - 如果存在 codebook 且未强制线性，则优先查表（codebook[total_bits][int_val]）
    - 否则退回到原来的线性反量化公式
    返回的是位于指定 device_（或当前全局 device）上的 0-D Tensor。
    诊断：若高 SNR 下任务准确率远低于预期（如 62% 而非 ~96%），可设环境变量
    FORCE_LINEAR_DEQUANT_IBP=1 强制使用线性反量化；若此时准确率恢复，则问题在码本（训练时
    未用码本 / 码本与编码端不一致 / 码本索引或数值范围异常）。
    """
    if device_ is None:
        device_ = device

    # 强制线性反量化（用于诊断：高 SNR 下若用码本准确率异常低，可先试线性是否恢复）
    if os.environ.get("FORCE_LINEAR_DEQUANT_IBP", "").strip() == "1":
        if total_bits <= 0:
            return torch.tensor(0.0, device=device_)
        M = float(2 ** total_bits)
        return torch.tensor((float(int_val) / M) * 2.0 - 1.0, device=device_)

    # 优先使用 codebook
    if CODEBOOKS is not None and total_bits in CODEBOOKS and total_bits > 0:
        cb = CODEBOOKS[total_bits]
        # 确保 codebook 在正确的 device 上
        if cb.device != device_:
            cb = cb.to(device_)
            CODEBOOKS[total_bits] = cb
        idx = int(int_val)
        if cb.numel() > 0:
            idx = max(0, min(idx, cb.numel() - 1))
            return cb[idx]

    # 回退：原来的线性反量化
    if total_bits <= 0:
        return torch.tensor(0.0, device=device_)
    M = float(2 ** total_bits)
    return torch.tensor((float(int_val) / M) * 2.0 - 1.0, device=device_)


# ==================== Utilities ====================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _noise_power_from_symbols(symbols: torch.Tensor, snr_db: float) -> float:
    """噪声功率 = 1/SNR。假定 QAM64/IBP/SOM/sDMCM 星座图均已能量归一化（平均符号能量=1），直接使用 1/snr_lin。"""
    snr_lin = 10 ** (snr_db / 10.0)
    return 1.0 / snr_lin


class AWGN(nn.Module):
    """AWGN：噪声功率 = 信号平均能量/SNR（已按接收平均能量计算）。"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, snr_db: float):
        if snr_db is None:
            return x
        power = x.pow(2).mean()
        snr_lin = 10 ** (snr_db / 10.0)
        noise_power = power / snr_lin
        noise_std = torch.sqrt(noise_power.clamp(min=1e-12))
        noise = torch.randn_like(x) * noise_std
        return x + noise


class QAM64:
    def __init__(self):
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        pts = []
        for i in levels:
            for q in levels:
                pts.append(complex(i, q))
        pts = np.array(pts)
        pts = pts / np.sqrt(np.mean(np.abs(pts) ** 2))
        self.constellation = torch.complex(
            torch.tensor(pts.real, dtype=torch.float32),
            torch.tensor(pts.imag, dtype=torch.float32),
        )
        self.bits_per_symbol = 6

    def modulate(self, bits: torch.Tensor) -> torch.Tensor:
        # bits: [B, Nbits] float in {0,1}
        B, N = bits.shape
        Ns = (N // self.bits_per_symbol)
        if Ns == 0:
            return torch.zeros(B, 0, dtype=torch.complex64, device=bits.device)
        bits = bits[:, : Ns * self.bits_per_symbol]
        bits = bits.view(B, Ns, self.bits_per_symbol)
        idx = torch.zeros(B, Ns, dtype=torch.long, device=bits.device)
        for i in range(self.bits_per_symbol):
            idx += (bits[:, :, i].long() << i)
        const = self.constellation.to(bits.device)
        syms = torch.zeros(B, Ns, dtype=torch.complex64, device=bits.device)
        for i in range(64):
            mask = (idx == i)
            if mask.any():
                syms[mask] = const[i]
        return syms

    def demodulate(self, symbols: torch.Tensor) -> torch.Tensor:
        # symbols: [B, Ns]
        if symbols.numel() == 0:
            return torch.zeros(symbols.size(0), 0, device=symbols.device)
        B, Ns = symbols.shape
        const = self.constellation.to(symbols.device)
        idx = torch.zeros(B, Ns, dtype=torch.long, device=symbols.device)
        ref = torch.stack([const for _ in range(B)])  # [B,64]
        for n in range(Ns):
            d2 = torch.abs(symbols[:, n].unsqueeze(1) - ref)  # [B,64]
            idx[:, n] = torch.argmin(d2, dim=1)
        bits = torch.zeros(B, Ns * self.bits_per_symbol, device=symbols.device)
        for i in range(self.bits_per_symbol):
            bits[:, i::self.bits_per_symbol] = ((idx >> i) & 1).float()
        return bits


class DitherQuantizer:
    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    @torch.jit.script
    def quantize_batch_jit(x: torch.Tensor, b: int, gamma: float) -> torch.Tensor:
        """
        JIT编译的批量量化函数
        x: [B, N] 特征矩阵
        b: 量化比特数
        gamma: 量化范围
        """
        if b <= 0:
            return torch.zeros_like(x)

        M = 2 ** b
        delta = 2 * gamma / M

        # 非减性均匀抖动量化
        dither = (torch.rand_like(x) - 0.5) * delta
        y = torch.clamp(x, -gamma, gamma)
        q = torch.floor((y + dither + gamma) / delta).clamp(0, M - 1)
        xq = q * delta - gamma + delta / 2
        return xq

    def quantize_batch(self, x: torch.Tensor, b: int) -> torch.Tensor:
        """批量量化，支持不同b值"""
        return self.quantize_batch_jit(x, b, self.gamma)

    def quantize_feature(self, x: torch.Tensor, b: int) -> torch.Tensor:
        """单特征量化，供 QAM 等调用；内部用 quantize_batch 处理"""
        if b <= 0:
            return torch.zeros_like(x)
        x_ = x.view(1, -1)  # 标量 -> (1,1)，向量 -> (1, N)
        return self.quantize_batch(x_, b).squeeze(0)

    def quantize_batch_multi_bits(self, x: torch.Tensor, bits_list: List[int]) -> torch.Tensor:
        """
        批量 QAM 量化，每样本可不同比特数。x: [N]，bits_list 长度 N。
        返回 int_values: [N] long。直接量化到整数，避免 float 中间结果与二次转换（与 IBP 一致）。
        """
        N = x.shape[0]
        if N == 0:
            return torch.zeros(0, dtype=torch.long, device=x.device)
        int_values = torch.zeros(N, dtype=torch.long, device=x.device)
        bit_configs = {}
        for i in range(N):
            b = bits_list[i]
            if b <= 0:
                continue
            if b not in bit_configs:
                bit_configs[b] = []
            bit_configs[b].append(i)
        for b, indices in bit_configs.items():
            idx = torch.as_tensor(np.array(indices, dtype=np.int64), device=x.device)
            x_sub = x[idx]
            M = 2 ** b
            delta = 2 * self.gamma / M
            x_clamped = x_sub.clamp(-self.gamma, self.gamma)
            dither = (torch.rand_like(x_sub, device=x.device) - 0.5) * delta
            q = torch.floor((x_clamped + dither + self.gamma) / delta).clamp(0, M - 1).long()
            int_values[idx] = q
        return int_values


class IBPQuantizer:
    """IBP量化器：使用非减性均匀抖动量化 + 重要/非重要比特分离；反量化由 dequantize_from_int 完成。"""

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def quantize_with_ibp(self, x: torch.Tensor, total_bits: int, imp_bits: int):
        """
        量化特征：使用非减性均匀抖动量化（non-subtractive uniform dither quantization）。
        返回 (quantized, imp_value, int_val_opt)：int_val_opt 为量化索引（用于编码传输），反量化用 dequantize_from_int。
        """
        if total_bits <= 0:
            return torch.zeros_like(x), None, None

        M_total = 2 ** total_bits
        nsk_bits = total_bits - imp_bits
        delta_total = 2 * self.gamma / M_total

        # 非减性均匀抖动量化：dither ~ U(-delta/2, delta/2)
        x_clamped = x.clamp(-self.gamma, self.gamma)
        dither = (torch.rand_like(x_clamped) - 0.5) * delta_total
        q = torch.floor((x_clamped + dither + self.gamma) / delta_total).clamp(0, M_total - 1)
        quantized = q * delta_total - self.gamma + delta_total / 2

        # 计算重要比特值（用于编码传输）
        int_value = q.long()
        if imp_bits > 0 and nsk_bits > 0:
            imp_value = int_value >> nsk_bits
        else:
            imp_value = int_value if imp_bits > 0 else torch.zeros_like(int_value)

        # int_val_opt 为量化索引（用于反量化）
        return quantized, imp_value, int_value

    def quantize_with_ibp_batch(self, x: torch.Tensor, total_bits_list: List[int], imp_bits_list: List[int]):
        """
        批量 IBP 量化，减少 GPU 调用次数。x: [N] 标量序列，total_bits_list/imp_bits_list 长度 N。
        返回 (int_values: [N] long, quantized: [N] float)。
        """
        N = x.shape[0]
        if N == 0:
            return torch.zeros(0, dtype=torch.long, device=x.device), torch.zeros(0, dtype=x.dtype, device=x.device)
        int_values = torch.zeros(N, dtype=torch.long, device=x.device)
        quantized_out = torch.zeros(N, dtype=x.dtype, device=x.device)
        bit_configs = {}
        for i in range(N):
            tb, ib = total_bits_list[i], imp_bits_list[i]
            if tb <= 0:
                continue
            key = (tb, ib)
            if key not in bit_configs:
                bit_configs[key] = []
            bit_configs[key].append(i)
        for (total_bits, imp_bits), indices in bit_configs.items():
            idx = torch.tensor(indices, dtype=torch.long, device=x.device)
            x_sub = x[idx]
            nsk_bits = total_bits - imp_bits
            M_total = 2 ** total_bits
            delta_total = 2 * self.gamma / M_total
            x_clamped = x_sub.clamp(-self.gamma, self.gamma)
            dither = (torch.rand_like(x_sub, device=x.device) - 0.5) * delta_total
            q = torch.floor((x_clamped + dither + self.gamma) / delta_total).clamp(0, M_total - 1)
            quantized_out[idx] = q * delta_total - self.gamma + delta_total / 2
            int_values[idx] = q.long()
        return int_values, quantized_out


class SOMModulator:
    """
    Semantic-Oriented Modulation (SOM) 调制解调器
    基于论文: Semantic-Oriented Modulation for Wireless Communication

    核心思想：
    - 直接将两个连续特征组合成复数：s_i = f_{2i} + j·f_{2i+1}
    - 使用L层M阶分层星座设计
    - 每层选择最近的星座点，然后放大残差到下一层
    - 解调时使用最小欧氏距离估计
    """

    def __init__(self, L=3, M=4, gamma=1.0):
        """
        Args:
            L: 层数（默认3层）
            M: 每层的调制阶数（默认4，即4-QAM）
            gamma: 特征范围（默认[-3, 3]）
        """
        self.L = L
        self.M = M
        self.gamma = gamma
        self.sqrt_M = math.sqrt(M)

        # 生成每层的星座点
        self.constellations = self._generate_constellations()

    def _generate_constellations(self):
        """生成L层M阶星座图

        根据论文，每层将星座平面分成M个区域，使用本地放大方式
        第1层：基础M-QAM星座，范围覆盖[-gamma, gamma]
        后续层：通过缩放因子1/√M生成更细的星座点
        """
        constellations = []

        # 第1层：基础M-QAM星座（范围覆盖[-gamma, gamma]）
        # 对于M=4，生成4-QAM星座点，中心在(-gamma/2, gamma/2)范围内
        if self.M == 4:
            # 4-QAM: 四个象限的中心点
            # 范围[-gamma, gamma]，所以中心点在±gamma/2
            base_const = torch.tensor([
                -self.gamma / 2 + 1j * self.gamma / 2,  # 第二象限
                self.gamma / 2 + 1j * self.gamma / 2,  # 第一象限
                -self.gamma / 2 - 1j * self.gamma / 2,  # 第三象限
                self.gamma / 2 - 1j * self.gamma / 2  # 第四象限
            ], dtype=torch.complex64)
        else:
            # 通用M-QAM生成（正方形星座）
            side = int(math.sqrt(self.M))
            if side * side != self.M:
                raise ValueError(f"M={self.M} 不是完全平方数，无法生成正方形星座")

            # 生成均匀分布的星座点
            step = self.gamma / (side - 1) if side > 1 else self.gamma
            levels = torch.linspace(-self.gamma / 2, self.gamma / 2, side)
            base_const = torch.zeros(self.M, dtype=torch.complex64)
            idx = 0
            for i in levels:
                for q in levels:
                    base_const[idx] = i + 1j * q
                    idx += 1

        constellations.append(base_const)

        # 后续层：通过缩放生成（每层缩小√M倍）
        for l in range(1, self.L):
            # 每层星座点缩小sqrt(M)倍
            scale = 1.0 / (self.sqrt_M ** l)
            layer_const = base_const * scale
            constellations.append(layer_const)

        # 统一能量归一化：按所有层所有点的平均功率归一到1
        all_points = torch.cat([c.reshape(-1) for c in constellations])
        power = (torch.abs(all_points) ** 2).mean()
        if power > 0:
            scale = 1.0 / torch.sqrt(power)
            constellations = [c * scale for c in constellations]

        return constellations

    def modulate(self, features: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
        """
        调制语义特征到符号序列

        Args:
            features: [B, N] 语义特征向量（N应该是偶数，因为每两个特征组成一个复数）
            snr_db: 信噪比（用于添加噪声）

        Returns:
            symbols: [B, N/2, L] 每对特征对应的L层符号序列
        """
        if not torch.is_tensor(features):
            features = torch.tensor(features)

        B, N = features.shape
        if N % 2 != 0:
            # 如果是奇数，补零
            features = torch.cat([features, torch.zeros(B, 1, device=features.device)], dim=1)
            N += 1

        num_pairs = N // 2
        symbols = torch.zeros(B, num_pairs, self.L, dtype=torch.complex64, device=features.device)

        # 将特征限制在[-gamma, gamma]范围内
        features = features.clamp(-self.gamma, self.gamma)

        for b in range(B):
            for i in range(num_pairs):
                # 组合两个特征成复数
                s = features[b, 2 * i] + 1j * features[b, 2 * i + 1]

                # 多层映射（根据论文的分层设计）
                residual = s
                for l in range(self.L):
                    # 选择最近的星座点
                    const = self.constellations[l].to(s.device)
                    distances = torch.abs(residual - const)
                    idx = torch.argmin(distances)
                    x_l = const[idx]
                    symbols[b, i, l] = x_l

                    # 计算残差并放大到下一层（本地放大方式）
                    # 残差 = (s - x_l) * √M，这样下一层可以进一步细化
                    if l < self.L - 1:
                        residual = (residual - x_l) * self.sqrt_M

        # 添加 AWGN：噪声功率 = 接收平均能量/SNR
        if snr_db is not None:
            noise_power = _noise_power_from_symbols(symbols, snr_db)
            noise_std = math.sqrt(noise_power / 2)
            noise_real = torch.randn_like(symbols.real, device=symbols.device) * noise_std
            noise_imag = torch.randn_like(symbols.imag, device=symbols.device) * noise_std
            symbols = symbols + torch.complex(noise_real, noise_imag)

        return symbols

    def demodulate(self, symbols: torch.Tensor) -> torch.Tensor:
        """
        解调符号序列到语义特征

        根据论文公式(13): ŝ_i = Σ_{l=1}^L (x̂_{i,l} / √M^{l-1})

        Args:
            symbols: [B, N/2, L] 接收到的符号序列

        Returns:
            features: [B, N] 恢复的语义特征向量
        """
        if not torch.is_tensor(symbols):
            symbols = torch.tensor(symbols)

        B, num_pairs, L = symbols.shape

        # 使用最小欧氏距离（MED）估计每层的符号
        recovered_complex = torch.zeros(B, num_pairs, dtype=torch.complex64, device=symbols.device)

        for b in range(B):
            for i in range(num_pairs):
                # 重构复数（根据论文公式）
                s_hat = torch.complex(torch.tensor(0.0, device=symbols.device),
                                      torch.tensor(0.0, device=symbols.device))

                for l in range(L):
                    # MED估计：选择最近的星座点
                    const = self.constellations[l].to(symbols.device)
                    received = symbols[b, i, l]
                    distances = torch.abs(received - const)
                    idx = torch.argmin(distances)
                    x_hat_l = const[idx]

                    # 累加（根据论文公式(13): ŝ_i = Σ_{l=1}^L (x̂_{i,l} / √M^{l-1})）
                    # l从0开始，所以第1层(l=0)缩放因子是1/√M^0=1，第2层(l=1)是1/√M^1，以此类推
                    scale_factor = 1.0 / (self.sqrt_M ** l)
                    s_hat = s_hat + x_hat_l * scale_factor

                recovered_complex[b, i] = s_hat

        # 从复数中提取实部和虚部作为特征
        features = torch.zeros(B, num_pairs * 2, device=symbols.device)
        features[:, 0::2] = recovered_complex.real
        features[:, 1::2] = recovered_complex.imag

        # 限制在合理范围内
        features = features.clamp(-self.gamma, self.gamma)

        return features


class SDMCMMapper:
    """
    sDMCM (Semantic Digital Modulation Constellation Mapping) 调制解调器
    基于论文: sDMCM—A Semantic Digital Modulation Constellation Mapping Scheme for Semantic Communication

    核心思想：
    - 将量化后的语义特征值直接映射到星座点，使得相邻星座点的数值差最小（而非比特差最小）
    - 保护高权重比特，减少语义失真
    - 对于 n=m 情况：XM = (2XQ + 1 - 2^m) * d
    - QAM 通过正交化 PAM 实现
    """

    def __init__(self, n_bits=4, m_bits=6, gamma=1.0):
        """
        Args:
            n_bits: 量化比特数（语义特征的量化精度）
            m_bits: 调制阶数（2^m_bits 阶 QAM，即每个符号 m_bits 比特）
            gamma: 特征范围 [-gamma, gamma]
        """
        self.n_bits = n_bits
        self.m_bits = m_bits
        self.gamma = gamma
        self.d = 1.0  # 相邻星座点距离的一半

        # 生成 sDMCM PAM 星座点
        self.pam_constellation = self._generate_pam_constellation()

        # 生成 sDMCM QAM 星座点（通过正交化 PAM）
        self.qam_constellation = self._generate_qam_constellation()

    def _generate_pam_constellation(self):
        """生成 sDMCM PAM 星座点（1D）"""
        M = 2 ** self.m_bits
        constellation = torch.zeros(M, dtype=torch.float32)

        # 根据论文公式(4): XM = (2XQ + 1 - 2^m) * d
        # 对于 n=m 情况，XQ 从 0 到 2^m-1
        for xq in range(M):
            xm = (2 * xq + 1 - M) * self.d
            constellation[xq] = xm

        # 归一化（保持平均功率为1）
        power = (constellation ** 2).mean()
        constellation = constellation / torch.sqrt(power)

        return constellation

    def _generate_qam_constellation(self):
        """生成 sDMCM QAM 星座点（2D，通过正交化 PAM）"""
        M = 2 ** self.m_bits
        pam = self.pam_constellation

        # QAM = PAM_I + j * PAM_Q（正交化）
        # 对于 2^(2m) 阶 QAM，需要两个 2^m 阶 PAM
        qam_const = torch.zeros(M * M, dtype=torch.complex64)
        idx = 0
        for i in range(M):
            for q in range(M):
                qam_const[idx] = torch.complex(pam[i], pam[q])
                idx += 1

        # 归一化
        power = (torch.abs(qam_const) ** 2).mean()
        qam_const = qam_const / torch.sqrt(power)

        return qam_const

    def _quantize_to_index(self, value: float) -> int:
        """将连续值量化到索引 [0, 2^n_bits-1]"""
        # 将值从 [-gamma, gamma] 映射到 [0, 2^n_bits-1]
        normalized = (value + self.gamma) / (2 * self.gamma)
        # 使用 max/min 限制范围（因为 normalized 是 float，不是 tensor）
        normalized = max(0.0, min(1.0, normalized))
        index = int(normalized * (2 ** self.n_bits - 1))
        return index

    def _index_to_value(self, index: int) -> float:
        """将索引转换为连续值"""
        # 从 [0, 2^n_bits-1] 映射回 [-gamma, gamma]
        normalized = index / (2 ** self.n_bits - 1) if self.n_bits > 0 else 0
        value = normalized * 2 * self.gamma - self.gamma
        return value

    def modulate(self, features: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
        """
        调制语义特征到符号

        Args:
            features: [B, N] 语义特征向量（N应该是偶数，因为每两个特征组成一个复数）
            snr_db: 信噪比

        Returns:
            symbols: [B, N/2] 调制后的复数符号
        """
        if not torch.is_tensor(features):
            features = torch.tensor(features)

        B, N = features.shape

        # 确保 N 是偶数（补零）
        if N % 2 != 0:
            features = torch.cat([features, torch.zeros(B, 1, device=features.device)], dim=1)
            N += 1

        num_pairs = N // 2

        # 将特征限制在 [-gamma, gamma] 范围内
        features = features.clamp(-self.gamma, self.gamma)

        symbols = torch.zeros(B, num_pairs, dtype=torch.complex64, device=features.device)
        qam_const = self.qam_constellation.to(features.device)
        M = 2 ** self.m_bits

        for b in range(B):
            for i in range(num_pairs):
                # 量化两个特征到索引
                idx_i = self._quantize_to_index(features[b, 2 * i].item())
                idx_q = self._quantize_to_index(features[b, 2 * i + 1].item())

                # 处理 n 和 m 的关系
                if self.n_bits == self.m_bits:
                    # 情况1: n = m，直接映射
                    idx_i_mapped = idx_i
                    idx_q_mapped = idx_q
                elif self.n_bits > self.m_bits:
                    # 情况2: n > m，取高 m_bits 位（保护MSB）
                    idx_i_mapped = idx_i >> (self.n_bits - self.m_bits)
                    idx_q_mapped = idx_q >> (self.n_bits - self.m_bits)
                else:
                    # 情况3: n < m，扩展到 m_bits（左移填充）
                    idx_i_mapped = idx_i << (self.m_bits - self.n_bits)
                    idx_q_mapped = idx_q << (self.m_bits - self.n_bits)

                # 限制在有效范围内
                idx_i_mapped = min(idx_i_mapped, M - 1)
                idx_q_mapped = min(idx_q_mapped, M - 1)

                # 映射到 QAM 星座点：QAM[i*M + q] = PAM[i] + j*PAM[q]
                qam_idx = idx_i_mapped * M + idx_q_mapped
                symbols[b, i] = qam_const[qam_idx]

        # 添加 AWGN：噪声功率 = 接收平均能量/SNR
        if snr_db is not None:
            noise_power = _noise_power_from_symbols(symbols, snr_db)
            noise_std = math.sqrt(noise_power / 2)
            noise_real = torch.randn_like(symbols.real, device=symbols.device) * noise_std
            noise_imag = torch.randn_like(symbols.imag, device=symbols.device) * noise_std
            noise = torch.complex(noise_real, noise_imag)
            symbols = symbols + noise

        return symbols

    def demodulate(self, symbols: torch.Tensor) -> torch.Tensor:
        """
        解调符号到语义特征

        Args:
            symbols: [B, N_symbols] 接收到的复数符号

        Returns:
            features: [B, 2*N_symbols] 恢复的语义特征向量
        """
        if not torch.is_tensor(symbols):
            symbols = torch.tensor(symbols)

        B, N_symbols = symbols.shape

        # 使用最小欧氏距离（MED）估计
        features = torch.zeros(B, 2 * N_symbols, device=symbols.device)
        qam_const = self.qam_constellation.to(symbols.device)
        pam_const = self.pam_constellation.to(symbols.device)
        M = 2 ** self.m_bits

        for b in range(B):
            for i in range(N_symbols):
                symbol = symbols[b, i]

                # 使用最小欧氏距离找到最近的 QAM 星座点
                distances = torch.abs(qam_const - symbol)
                qam_idx = torch.argmin(distances).item()

                # 从 QAM 索引恢复 I 和 Q 的索引
                idx_i_mapped = qam_idx // M
                idx_q_mapped = qam_idx % M

                # 处理 n 和 m 的关系（反向映射）
                if self.n_bits == self.m_bits:
                    idx_i = idx_i_mapped
                    idx_q = idx_q_mapped
                elif self.n_bits > self.m_bits:
                    # 情况2: n > m，恢复时左移（低 n-m 位补0）
                    idx_i = idx_i_mapped << (self.n_bits - self.m_bits)
                    idx_q = idx_q_mapped << (self.n_bits - self.m_bits)
                else:
                    # 情况3: n < m，恢复时右移（丢弃低 m-n 位）
                    idx_i = idx_i_mapped >> (self.m_bits - self.n_bits)
                    idx_q = idx_q_mapped >> (self.m_bits - self.n_bits)

                # 限制在有效范围内
                max_idx = 2 ** self.n_bits - 1
                idx_i = min(idx_i, max_idx)
                idx_q = min(idx_q, max_idx)

                # 将索引转换回连续值
                value_i = self._index_to_value(idx_i)
                value_q = self._index_to_value(idx_q)

                features[b, 2 * i] = value_i
                features[b, 2 * i + 1] = value_q

        # 限制在合理范围内
        features = features.clamp(-self.gamma, self.gamma)

        return features


# ==================== JSCC (Joint Source-Channel Coding) 基准方案 ====================
"""
JSCC 基准方案 - 语义通信论文常用基准
参考: Deep Joint Source-Channel Coding for Semantic Communications (Bourtsoulas et al., 2019)
      DeepJSCC: 端到端联合源信道编码，信源信道联合训练

约束: 仅一对编解码器，对接处（输出口）仅166个复符号 ≈ 1000比特等效
流程: 图像 → JSCCEncoder → 166复符号(332实维) → AWGN → JSCCDecoder → 10类
"""
JSCC_N_SYMBOLS = 166  # 信道接口符号数（166复符号 = 332实维 ≈ 996比特）


class JSCCEncoder(nn.Module):
    """
    JSCC 编码器：图像 → 166个复符号（332维实数）。
    唯一输出口为166符号，功率归一化使平均符号能量=1。
    """

    def __init__(self, n_symbols: int = JSCC_N_SYMBOLS):
        super().__init__()
        self.n_symbols = n_symbols
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_symbols * 2),  # 166*2 = 332 (I,Q)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, 224, 224] 图像
        返回: [B, 332] 166复符号的实虚部，功率归一化
        """
        out = self.backbone(x)  # [B, 332]
        power = (out ** 2).mean(dim=1, keepdim=True).clamp(min=1e-12)
        out = out / torch.sqrt(power)
        return out


class JSCCDecoder(nn.Module):
    """
    JSCC 解码器：166个复符号（332维）→ 10类。
    唯一输入口为166符号。
    """

    def __init__(self, n_symbols: int = JSCC_N_SYMBOLS, num_classes: int = 10):
        super().__init__()
        self.n_symbols = n_symbols
        self.net = nn.Sequential(
            nn.Linear(n_symbols * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 332] 接收的166复符号（实虚部）
        返回: [B, num_classes] 分类 logits
        """
        return self.net(x)


def _jscc_awgn_complex(symbols: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    对复符号添加AWGN。symbols: [B, 332] 实虚部拼接。
    噪声功率 = 信号功率/SNR，与 _noise_power_from_symbols 一致。
    """
    if snr_db is None:
        return symbols
    B = symbols.shape[0]
    # 将 [B, 332] 视为 [B, 166] 复数
    power = (symbols ** 2).mean()
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = power / snr_lin
    noise_std = math.sqrt(noise_power / 2)  # 实部虚部各一半
    noise = torch.randn_like(symbols, device=symbols.device) * noise_std
    return symbols + noise


# ==================== IBP 双流打包：重要比特/非重要比特分离，确保 2 重要 + 4 非重要/符号 ====================
# 每符号 6 比特：位置 0,1 = 重要比特（低 SER），位置 2,3,4,5 = 非重要比特（高 SER）
IBP_IMP_PER_SYMBOL = 2
IBP_NSK_PER_SYMBOL = 4


def pack_ibp_dual_stream(imp_stream: List[int], nsk_stream: List[int], device=None) -> torch.Tensor:
    """
    将重要比特流和非重要比特流打包为 IBP 符号序列。每符号 = 2 重要 + 4 非重要，不足补 0。
    返回 [1, num_symbols*6] 的 bits tensor。向量化实现，避免 Python 位级循环。
    """
    num_symbols = max(
        (len(imp_stream) + IBP_IMP_PER_SYMBOL - 1) // IBP_IMP_PER_SYMBOL,
        (len(nsk_stream) + IBP_NSK_PER_SYMBOL - 1) // IBP_NSK_PER_SYMBOL,
        1 if (imp_stream or nsk_stream) else 0,
    )
    imp_arr = np.array(imp_stream, dtype=np.float32) if imp_stream else np.zeros(0, dtype=np.float32)
    nsk_arr = np.array(nsk_stream, dtype=np.float32) if nsk_stream else np.zeros(0, dtype=np.float32)
    imp_pad = np.zeros(num_symbols * IBP_IMP_PER_SYMBOL, dtype=np.float32)
    nsk_pad = np.zeros(num_symbols * IBP_NSK_PER_SYMBOL, dtype=np.float32)
    imp_pad[:min(len(imp_arr), len(imp_pad))] = imp_arr[:len(imp_pad)]
    nsk_pad[:min(len(nsk_arr), len(nsk_pad))] = nsk_arr[:len(nsk_pad)]
    imp_2d = imp_pad.reshape(-1, IBP_IMP_PER_SYMBOL)
    nsk_2d = nsk_pad.reshape(-1, IBP_NSK_PER_SYMBOL)
    interleaved = np.concatenate([imp_2d, nsk_2d], axis=1).flatten()
    t = torch.from_numpy(interleaved.astype(np.float32))
    if device is not None:
        t = t.to(device)
    return t.unsqueeze(0)


def pack_ibp_dual_stream_batch(imp_stream_list: List[List[int]], nsk_stream_list: List[List[int]],
                               device=None) -> torch.Tensor:
    """批量打包 B 个双流为 [B, pad_to] 张量，向量化实现。"""
    B = len(imp_stream_list)
    if B == 0:
        return torch.zeros(0, 0, dtype=torch.float32, device=device)
    tensors = [pack_ibp_dual_stream(imp_stream_list[b], nsk_stream_list[b], device=device).squeeze(0) for b in range(B)]
    max_len = max(t.shape[0] for t in tensors)
    pad_to = ((max_len + 5) // 6) * 6
    padded = []
    for t in tensors:
        if t.shape[0] < pad_to:
            t = torch.cat([t, torch.zeros(pad_to - t.shape[0], device=t.device, dtype=t.dtype)])
        padded.append(t)
    return torch.stack(padded)


def unpack_ibp_dual_stream(demod_bits: torch.Tensor, bit_allocation: List[Tuple], batch_idx: int = 0,
                           device=None) -> List[Tuple[int, int, int]]:
    """
    从解调比特中按双流格式提取，按 bit_allocation 归还给各特征。
    demod_bits: [B, N]，每 6 比特 = 2 重要 + 4 非重要
    bit_allocation: [(feat_idx, total_bits, imp_bits, nsk_bits), ...]
    返回 [(feat_idx, int_val, total_bits), ...] 用于 dequantize_from_int。向量化实现。
    """
    if device is None:
        device = demod_bits.device
    bits = demod_bits[batch_idx].cpu().numpy()
    num_symbols = bits.shape[0] // 6
    if num_symbols == 0:
        return []
    bits_2d = bits[:num_symbols * 6].reshape(num_symbols, 6)
    imp_stream = (bits_2d[:, :IBP_IMP_PER_SYMBOL].flatten() > 0.5).astype(np.uint8)
    nsk_stream = (bits_2d[:, IBP_IMP_PER_SYMBOL:].flatten() > 0.5).astype(np.uint8)
    imp_idx, nsk_idx = 0, 0
    result = []
    for alloc in bit_allocation:
        feat_idx, total_bits, imp_bits, nsk_bits = alloc[0], alloc[1], alloc[2], alloc[3]
        imp_slice = imp_stream[imp_idx:imp_idx + imp_bits] if imp_bits > 0 else np.zeros(0, dtype=np.uint8)
        nsk_slice = nsk_stream[nsk_idx:nsk_idx + nsk_bits] if nsk_bits > 0 else np.zeros(0, dtype=np.uint8)
        if len(imp_slice) < imp_bits:
            imp_slice = np.pad(imp_slice, (0, imp_bits - len(imp_slice)), constant_values=0)
        if len(nsk_slice) < nsk_bits:
            nsk_slice = np.pad(nsk_slice, (0, nsk_bits - len(nsk_slice)), constant_values=0)
        imp_val = int(np.dot(imp_slice, 2 ** np.arange(imp_bits - 1, -1, -1))) if imp_bits > 0 else 0
        nsk_val = int(np.dot(nsk_slice, 2 ** np.arange(nsk_bits - 1, -1, -1))) if nsk_bits > 0 else 0
        int_val = (imp_val << nsk_bits) | nsk_val
        result.append((feat_idx, int_val, total_bits))
        imp_idx += imp_bits
        nsk_idx += nsk_bits
    return result


def pack_qam_bits_batch(all_bits_list: List[List], device=None) -> torch.Tensor:
    """批量打包 B 个 QAM 比特流为 [B, pad_to] 张量，向量化实现。QAM 为单流（无 imp/nsk 拆分）。"""
    B = len(all_bits_list)
    if B == 0:
        return torch.zeros(0, 0, dtype=torch.float32, device=device)
    max_len = max(len(lst) for lst in all_bits_list) if any(all_bits_list) else 0
    if max_len == 0:
        return torch.zeros(B, 6, dtype=torch.float32, device=device)
    pad_to = ((max_len + 5) // 6) * 6
    arr = np.zeros((B, pad_to), dtype=np.float32)
    for b in range(B):
        if all_bits_list[b]:
            arr[b, :len(all_bits_list[b])] = np.array(all_bits_list[b], dtype=np.float32)
    t = torch.from_numpy(arr)
    if device is not None:
        t = t.to(device)
    return t


# ==================== IBP 3 层星座（定坐标表，alpha1/alpha2 相对子星座中心放缩）====================
# 替换原 IBP 方案：alpha1 等效原 alpha，alpha2 默认 1（暂不处理第三层）
_IBP_CONSTELLATION_TABLE = [
    ("000000", -7, -7), ("000010", -5, -7), ("001010", -3, -7), ("001000", -1, -7), ("101000", 1, -7),
    ("101010", 3, -7), ("100010", 5, -7), ("100000", 7, -7),
    ("000001", -7, -5), ("000011", -5, -5), ("001011", -3, -5), ("001001", -1, -5), ("101001", 1, -5),
    ("101011", 3, -5), ("100011", 5, -5), ("100001", 7, -5),
    ("000101", -7, -3), ("000111", -5, -3), ("001111", -3, -3), ("001101", -1, -3), ("101101", 1, -3),
    ("101111", 3, -3), ("100111", 5, -3), ("100101", 7, -3),
    ("000100", -7, -1), ("000110", -5, -1), ("001110", -3, -1), ("001100", -1, -1), ("101100", 1, -1),
    ("101110", 3, -1), ("100110", 5, -1), ("100100", 7, -1),
    ("010100", -7, 1), ("010110", -5, 1), ("011110", -3, 1), ("011100", -1, 1), ("111100", 1, 1), ("111110", 3, 1),
    ("110110", 5, 1), ("110100", 7, 1),
    ("010101", -7, 3), ("010111", -5, 3), ("011111", -3, 3), ("011101", -1, 3), ("111101", 1, 3), ("111111", 3, 3),
    ("110111", 5, 3), ("110101", 7, 3),
    ("010001", -7, 5), ("010011", -5, 5), ("011011", -3, 5), ("011001", -1, 5), ("111001", 1, 5), ("111011", 3, 5),
    ("110011", 5, 5), ("110001", 7, 5),
    ("010000", -7, 7), ("010010", -5, 7), ("011010", -3, 7), ("011000", -1, 7), ("111000", 1, 7), ("111010", 3, 7),
    ("110010", 5, 7), ("110000", 7, 7),
]


def _ibp_sub_center_layer1(I, Q):
    """层1 子星座中心：4 象限中心。"""
    ci = 4.0 if I >= 0 else -4.0
    cq = 4.0 if Q >= 0 else -4.0
    return (ci, cq)


def _ibp_sub_center_layer2(I, Q):
    """层2 子星座中心：2×2 块中心。"""
    level_centers = (-6.0, -2.0, 2.0, 6.0)
    idx_i = max(0, min(3, (int(I) + 7) // 4))
    idx_q = max(0, min(3, (int(Q) + 7) // 4))
    return (level_centers[idx_i], level_centers[idx_q])


class IBPMapper:
    """IBP 星座：3 层定坐标表，统一使用 alpha=0.4（与 QAM 一致，4dB 训练、全 SNR 评估）。"""

    def __init__(self, total_bits=6, imp_bits=2, alpha=0.4, optimal_alpha_data=None, use_fixed_alpha=True):
        self.total_bits = total_bits
        self.imp_bits = imp_bits
        self.nsk_bits = total_bits - imp_bits
        self.alpha = alpha
        self._use_fixed_alpha = use_fixed_alpha
        # 固定 alpha=0.4：不再按 SNR 加载最优 alpha
        self._build_constellation(alpha)

    def _build_constellation(self, alpha):
        """根据 alpha 重建星座（bit_to_constellation、constellation、bit_strings_list），并构建向量化查表。"""
        self.alpha = alpha
        alpha1 = alpha
        alpha2 = 1.0
        self.bit_to_constellation = {}
        self.constellation_to_bit = {}
        self.constellation = []
        self.bit_strings_list = []
        for bits, I, Q in _IBP_CONSTELLATION_TABLE:
            c1_i, c1_q = _ibp_sub_center_layer1(I, Q)
            c2_i, c2_q = _ibp_sub_center_layer2(I, Q)
            pt_i = c1_i + alpha1 * (c2_i - c1_i) + alpha2 * (I - c2_i)
            pt_q = c1_q + alpha1 * (c2_q - c1_q) + alpha2 * (Q - c2_q)
            point = pt_i + 1j * pt_q
            self.bit_to_constellation[bits] = point
            self.constellation_to_bit[point] = bits
            self.constellation.append(point)
            self.bit_strings_list.append(bits)
        self.constellation = torch.tensor(self.constellation, dtype=torch.complex64)
        power = (torch.abs(self.constellation) ** 2).mean()
        if power > 0:
            self.constellation = self.constellation / torch.sqrt(power)
        # 向量化调制/解调用查表：线性索引 0..63 (6 比特) -> 星座下标；星座下标 -> 6 比特
        bit_string_to_idx = {s: i for i, s in enumerate(self.bit_strings_list)}
        self._lut_linear_to_const_idx = torch.tensor(
            [bit_string_to_idx.get("{:06b}".format(l), 0) for l in range(64)], dtype=torch.long
        )
        self._bits_lut = torch.tensor(
            [[int(b) for b in self.bit_strings_list[i]] for i in range(64)], dtype=torch.float32
        )

    def _ensure_alpha_for_snr(self, snr_db: float):
        """固定 alpha=0.4，不随 SNR 变化。"""
        pass

    def modulate(self, bits: torch.Tensor, snr_db: float = 10) -> torch.Tensor:
        """调制：每符号 6 比特 -> IBP 符号（向量化查表，无循环/无 .item()）。固定 alpha=0.4。"""
        self._ensure_alpha_for_snr(snr_db)
        if not torch.is_tensor(bits):
            bits = torch.tensor(bits, device=self.constellation.device)
        batch_size = bits.shape[0]
        bits_per_symbol = self.total_bits
        total_bits = bits.shape[1]
        num_symbols = total_bits // bits_per_symbol
        if num_symbols == 0:
            return torch.zeros(batch_size, 1, dtype=torch.complex64, device=bits.device)
        bits = bits[:, :num_symbols * bits_per_symbol].view(batch_size, num_symbols, bits_per_symbol)
        # 6 比特 -> 线性索引 0..63（左为高位，与 bit_strings_list 的 "000000" 顺序一致）
        bits_long = (bits > 0.5).long().to(bits.device)
        powers = torch.tensor([32, 16, 8, 4, 2, 1], device=bits.device, dtype=torch.long)
        linear = (bits_long * powers).sum(dim=-1)
        linear = linear.clamp(0, 63)
        lut = self._lut_linear_to_const_idx.to(bits.device)
        const_idx = lut[linear]
        const = self.constellation.to(bits.device)
        symbols_tensor = const[const_idx]
        if snr_db is None:
            return symbols_tensor
        return self._add_noise(symbols_tensor, snr_db)

    def demodulate(self, symbols: torch.Tensor) -> torch.Tensor:
        """硬判决解调（向量化：一次距离矩阵 + argmin + 查表，无循环/无 .item()）。"""
        if not torch.is_tensor(symbols):
            symbols = torch.tensor(symbols)
        batch_shape = symbols.shape
        batch_size = batch_shape[0]
        num_symbols = batch_shape[1] if len(batch_shape) > 1 else 1
        if batch_size == 0 or num_symbols == 0:
            return torch.zeros(batch_size, 0, device=symbols.device, dtype=torch.float32)
        constellation_tensor = self.constellation.to(symbols.device)
        # [B, N] vs [64] -> 扩展为 [B, N, 64]，沿最后一维取 argmin 得 [B, N]
        dist = torch.abs(symbols.unsqueeze(-1) - constellation_tensor)
        idx = dist.argmin(dim=-1).clamp(0, 63)
        bits_lut = self._bits_lut.to(symbols.device)
        bits_out = bits_lut[idx]
        return bits_out.reshape(batch_size, -1)

    def _add_noise(self, symbols: torch.Tensor, snr_db: float):
        """添加 AWGN。噪声功率 = 接收平均能量/SNR。"""
        if snr_db is None:
            return symbols
        noise_power = _noise_power_from_symbols(symbols, snr_db)
        n_real = torch.randn_like(symbols.real, device=symbols.device) * math.sqrt(noise_power / 2)
        n_imag = torch.randn_like(symbols.imag, device=symbols.device) * math.sqrt(noise_power / 2)
        return symbols + torch.complex(n_real, n_imag)


# ==================== Models ====================
class SemanticEncoder(nn.Module):
    def __init__(self, output_dim: int = 512):
        super().__init__()
        self.backbone = models.resnet18(pretrained=True)
        # 按论文设置 Dropout=0.3（应用于分类头之前）
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class SemanticDecoder(nn.Module):
    def __init__(self, input_dim: int = 512, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==================== Importance (STR + ISR) ====================
@torch.no_grad()
def compute_isr(A: torch.Tensor) -> torch.Tensor:
    """
    Inter-Semantics Relevance (ISR) - 严格按照论文公式(14)-(15)

    论文公式(15): ISR_k = (1/(C-1)) * Σ_{j≠k} sim(A_k, A_j)

    其中：
    - A_k: 第k个语义维度（在batch上的向量 A[:, k] ∈ R^B）
    - sim(A_k, A_j): 语义维度k和j之间的余弦相似度

    Args:
        A: [B, C] 语义特征矩阵，B=batch_size, C=特征维度(512)

    Returns:
        ISR: [C] 每个语义维度的重要性
    """
    # A: [B, 512]
    if A.dim() != 2:
        raise ValueError(f"compute_isr expects A to be 2D [B,C], got shape={tuple(A.shape)}")
    B, C = A.shape
    if C <= 1:
        return torch.zeros(C, device=A.device, dtype=A.dtype)

    # 转置：每行是一个语义维度 A_k ∈ R^B
    X = A.t().contiguous()  # [C, B]
    X = F.normalize(X, p=2, dim=1)  # 归一化每个语义维度

    # 计算语义维度间的相似度矩阵
    sim_mat = X @ X.t()  # [C, C], sim(A_k, A_j)
    sim_mat.fill_diagonal_(0.0)  # 对角线置0（排除自身）

    # 按照论文公式(15)：每个维度与其他维度的平均相似度
    v = sim_mat.sum(dim=1) / (C - 1)  # [C]
    return v


def compute_str_per_sample(encoder: nn.Module, decoder: nn.Module, images: torch.Tensor,
                           labels: torch.Tensor) -> torch.Tensor:
    """STR 逐样本版本，返回 [B, 512]，用于每图语义重要性。"""
    encoder.eval();
    decoder.eval()
    images.requires_grad_(True)
    feats = encoder(images)
    feats.retain_grad()
    logits = decoder(feats)
    sel = logits.gather(1, labels.view(-1, 1)).sum()
    sel.backward()
    grads = feats.grad
    if grads is None:
        grads = feats.detach()
    return grads.abs()


def compute_features_per_image(
        encoder: nn.Module, dataset, batch_size: int = 64,
) -> torch.Tensor:
    """
    预计算每张图的语义特征 [N, 512]，按数据集顺序。
    与 omega 类似，可提前生成并保存，训练/评估时按 indices 取用，避免重复调用 encoder。
    """
    encoder.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=(device.type == "cuda"))
    all_feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Features per image"):
            if len(batch) == 3:
                images = batch[0]
            else:
                images = batch[0]
            images = images.to(device)
            if device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    feats = encoder(images)
                feats = feats.float()
            else:
                feats = encoder(images)
            all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


def compute_omega_per_image(
        encoder: nn.Module, decoder: nn.Module, train_dataset, batch_size: int = 64, max_batches: Optional[int] = None
) -> torch.Tensor:
    """
    计算每张训练图的语义重要性 omega，形状 [N_train, 512]。
    STR 逐样本，ISR 在全体上平均；omega_i = STR_i * ISR_global，再按行 min-max 归一化。
    内部强制 cuDNN 确定性，使结果与随机种子无关、可复现。
    """
    cudnn_deterministic_orig = torch.backends.cudnn.deterministic
    cudnn_benchmark_orig = torch.backends.cudnn.benchmark
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return _compute_omega_per_image_impl(encoder, decoder, train_dataset, batch_size, max_batches)
    finally:
        torch.backends.cudnn.deterministic = cudnn_deterministic_orig
        torch.backends.cudnn.benchmark = cudnn_benchmark_orig


def _compute_omega_per_image_impl(
        encoder: nn.Module, decoder: nn.Module, train_dataset, batch_size: int = 64, max_batches: Optional[int] = None
) -> torch.Tensor:
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=(device.type == "cuda"))
    all_str = []
    all_isr = []
    count = 0
    for images, labels in tqdm(loader, desc="Omega per image"):
        images, labels = images.to(device), labels.to(device)
        str_w = compute_str_per_sample(encoder, decoder, images, labels)
        isr_w = compute_isr(encoder(images).detach())
        all_str.append(str_w)
        all_isr.append(isr_w)
        count += 1
        if max_batches is not None and count >= max_batches:
            break
    STR_all = torch.cat(all_str, dim=0)
    ISR = torch.stack(all_isr, dim=0).mean(dim=0)
    omega_raw = STR_all * ISR
    w_min = omega_raw.min(dim=1, keepdim=True)[0]
    w_max = omega_raw.max(dim=1, keepdim=True)[0]
    omega = (omega_raw - w_min) / (w_max - w_min + 1e-8)
    # 论文 Fig.11：重要性归一化为 sum=1，使 Eq.(10) 的 d 为加权平均 MSE（尺度合理）
    omega = omega / (omega.sum(dim=1, keepdim=True) + 1e-8)
    return omega.detach()


# ==================== PPO (discrete dynamic actions) ====================
# 论文仿真参数对照：η=0.99, δ=1e-3, β=0.5, ε=0.25, c₁=0.5, c₂=0.01, L₀=10, E_p=20
# 论文 Fig.11 要求 ω 和为 1，故 d 为加权平均 MSE、尺度小；L0=10 与 Table I 一致。
@dataclass
class PPOConfig:
    state_dim: int = 1 + 512 + 3  # a_i + omega(512) + [rem/total, rem_imp/init_imp, rem_nsk/init_nsk]
    action_max_per_step: int = 8
    learning_rate: float = 1e-3  # 论文 δ=1×10⁻³
    gamma: float = 0.99  # 论文 discount factor η=0.99
    gae_lambda: float = 0.95  # GAE λ（论文未单独给出）
    clip_epsilon: float = 0.25  # 论文 ε=0.25
    # 论文 Eq.(20)：c1 为 value loss 权重、c2 为 entropy bonus 权重
    value_coef: float = 0.05  # 论文 c₁=0.5
    entropy_coef: float = 0.001  # 论文 c₂=0.01
    update_epochs: int =1  # 每次 update() 内梯度轮数（原10，减半以加速训练）
    batch_size: int = 128
    minibatch: int = 64  # 单次 update() 内每个 minibatch 的样本数；小 batch 时避免 minibatch 过大
    max_grad_norm: float = 0.5
    use_ibp: bool = False  # 是否使用IBP模式
    action_space_size_ibp: int = None  # IBP 动作数（含(0,0)），为 None 则用 (max+1)*(max+2)//2
    L0: float = 10.0  # 奖励常数项（论文 Table I；差分奖励时仅在最后一步加）
    beta: float = 0.5  # 失真权重系数（论文 Table I：β=0.5）
    # 每步奖励间隔：1=每步都算（论文，效果有保证）；>1 为近似加速
    step_reward_interval: int = 1
    # 双 GPU 分段奖励已禁用：每 batch 内 deepcopy(decoder) 等导致 CPU 虚高、显存碎片，改为单卡向量化
    use_two_gpu_partial_reward: bool = False
    # 批量部分奖励时 decoder 在双 GPU 上拆分并行；单 GPU 或内存紧张时建议 False
    use_dual_gpu_decoder: bool = False
    # 语义重要性图选择开关：训练/评估是否使用 per-image omega（否则使用全局 omega）
    use_per_image_omega_train: bool = True
    use_per_image_omega_eval: bool = True
    # 是否对 value loss 做尺度归一化：adv 已归一化，ret 尺度可达 500–3500，不归一化易导致 Critic 梯度过大、难以收敛
    normalize_value_target: bool = False
    # 是否使用差分奖励：r_k = (Lk - β·dk) - (L_{k-1} - β·d_{k-1})，避免绝对效用导致的负分累积
    use_difference_reward: bool = False
    # IBP alpha EMA 更新：tau 大则 alpha 更快逼近 opt_alpha（0.2≈5次更新达63%，0.05需20次）
    alpha_ema_tau: float = 0.1
    # IBP alpha 更新间隔：1=每迭代更新（compute_optimal_alpha 已优化，每 iter 可接受）
    alpha_update_interval: int = 5
    # IBP alpha 解析优化：NSV（406 闭式注水 + 320/406 的 W1 与 Sigma 特征分解）。True 时传入 feats_batch+decoder；False 时仅 omega
    alpha_use_nsv_weights: bool = True
    alpha_nsv_bit_coupling: float = 0.3  # γ：每特征分配比特 total_bits 对 alpha 目标的增益系数
    # "406"：PCA + h_k=||W1 q_k||^2 + get_nsv 注水功率映回坐标；"diag"：仅用 diag(W^T W)*列方差
    alpha_nsv_weight_scheme: str = "406"
    # 课程学习：True 时在 -6 到 4 dB（步长2）上均分训练步数
    use_curriculum: bool = True
    # 内存紧张时：更频繁 empty_cache、减小 images_per_round、禁用双 GPU decoder
    low_memory_mode: bool = True
    # === 比特使用行为约束（关键：避免大量 0-bit 导致预算未用完）===
    # 每步选择 0 比特时的惩罚（仅在 remaining>0 时生效）；越大越逼迫“有预算就分配”
    zero_bit_penalty: float = 0.2
    # episode 结束时剩余比特的惩罚（按 remaining/total_budget 线性）；越大越逼迫“用满 1000bit”
    leftover_bit_penalty: float = 0.4


def _ppo_checkpoint_path(mode: str, action_max_per_step: int) -> str:
    """按模式与每特征最大比特数区分 checkpoint，避免 6-bit 模型被 10-bit 误加载。"""
    return os.path.abspath(f"ppo_ckpt_{str(mode).lower()}_max{int(action_max_per_step)}.pth")


def _save_ppo_checkpoint(path: str, ppo: "PPOAgent", *,
                         global_iteration: int,
                         training_rewards: Optional[list] = None,
                         extra: Optional[dict] = None) -> None:
    """保存可续训的完整 DPPO 状态（含 optimizer、随机状态、cfg 等）。"""
    ckpt = {
        "version": 1,
        "mode": "ibp" if getattr(ppo.cfg, "use_ibp", False) else "qam",
        "cfg": asdict(ppo.cfg),
        "global_iteration": int(global_iteration),
        "actor": ppo.actor.state_dict(),
        "critic": ppo.critic.state_dict(),
        "opt_act": ppo.opt_act.state_dict() if hasattr(ppo, "opt_act") and ppo.opt_act is not None else None,
        "opt_cri": ppo.opt_cri.state_dict() if hasattr(ppo, "opt_cri") and ppo.opt_cri is not None else None,
        "training_rewards": list(training_rewards) if training_rewards is not None else list(getattr(ppo, "training_rewards", [])),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        },
        "extra": extra or {},
    }
    torch.save(ckpt, path)


def _try_load_ppo_checkpoint(path: str, ppo: "PPOAgent") -> Tuple[bool, int]:
    """尝试加载完整 checkpoint。返回 (loaded_ok, global_iteration)。"""
    if not os.path.exists(path):
        return False, 0
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "actor" not in data or "critic" not in data:
        return False, 0
    # 模型/优化器
    ppo.actor.load_state_dict(data["actor"], strict=True)
    ppo.critic.load_state_dict(data["critic"], strict=True)
    if data.get("opt_act") is not None:
        try:
            ppo.opt_act.load_state_dict(data["opt_act"])
        except Exception:
            pass
    if data.get("opt_cri") is not None:
        try:
            ppo.opt_cri.load_state_dict(data["opt_cri"])
        except Exception:
            pass
    # rewards
    if "training_rewards" in data:
        try:
            ppo.training_rewards = list(data["training_rewards"])
        except Exception:
            pass
    # RNG（尽量恢复；缺字段则忽略）
    rng = data.get("rng", {}) if isinstance(data.get("rng", {}), dict) else {}
    try:
        if rng.get("python") is not None:
            random.setstate(rng["python"])
    except Exception:
        pass
    try:
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
    except Exception:
        pass
    try:
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
    except Exception:
        pass
    try:
        if torch.cuda.is_available() and rng.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])
    except Exception:
        pass
    it = int(data.get("global_iteration", 0) or 0)
    return True, it


class Actor(nn.Module):
    def __init__(self, state_dim: int, max_actions: int, use_ibp: bool = False, action_space_size_ibp: int = None):
        super().__init__()
        self.max_actions = max_actions
        self.use_ibp = use_ibp
        if use_ibp:
            # IBP: 含0比特时为 (max_actions+1)*(max_actions+2)//2；排除0比特时传 action_space_size_ibp
            action_space_size = (action_space_size_ibp if action_space_size_ibp is not None
                                 else (max_actions + 1) * (max_actions + 2) // 2)
            self.action_space_size = action_space_size
            self.net = nn.Sequential(
                nn.Linear(state_dim, 100), nn.Tanh(),
                nn.Linear(100, 64), nn.Tanh(),
                nn.Linear(64, action_space_size),  # 动态计算的动作空间大小
            )
        else:
            # QAM: 0到max_actions-1比特，共max_actions个动作（含0比特）
            self.net = nn.Sequential(
                nn.Linear(state_dim, 100), nn.Tanh(),
                nn.Linear(100, 64), nn.Tanh(),
                nn.Linear(64, max_actions),  # 0到max_actions-1比特
            )

    def forward(self, s: torch.Tensor, dynamic_size: int = None) -> torch.distributions.Categorical:
        logits = self.net(s)
        if not self.use_ibp and dynamic_size is not None:
            # QAM: dynamic_size是1到max_bits，动作索引k对应k+1比特
            logits = logits[:, :dynamic_size]  # 共dynamic_size个动作（1..dynamic_size比特）
        return torch.distributions.Categorical(logits=logits)


class Critic(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 100), nn.Tanh(),
            nn.Linear(100, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


# 在训练循环中添加内存优化
class MemoryOptimizer:
    """内存优化管理器"""

    def __init__(self, clear_freq=10):
        self.clear_freq = clear_freq
        self.counter = 0

    def step(self, iteration: int):
        """每N步执行一次深度清理"""
        self.counter += 1
        if self.counter % self.clear_freq == 0:
            # 清理所有可能的内存泄漏
            self.deep_clean()

    def deep_clean(self):
        """深度清理"""
        # 清理所有中间张量
        objects = gc.get_objects()
        for obj in objects:
            if torch.is_tensor(obj):
                # 清理不在使用的中间张量
                if obj.grad is not None:
                    obj.grad = None

        # 强制垃圾回收
        gc.collect()

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 清理CuDNN缓存
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True


class PPOAgent:
    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        as_ibp = getattr(cfg, "action_space_size_ibp", None)
        # QAM含0比特时需 action_max_per_step+1 个输出；IBP 用 action_space_size_ibp
        max_actions = (cfg.action_max_per_step + 1) if not cfg.use_ibp else cfg.action_max_per_step
        self.actor = Actor(cfg.state_dim, max_actions, cfg.use_ibp, action_space_size_ibp=as_ibp).to(device)
        self.critic = Critic(cfg.state_dim).to(device)
        self.opt_act = optim.Adam(self.actor.parameters(), lr=cfg.learning_rate)
        self.opt_cri = optim.Adam(self.critic.parameters(), lr=cfg.learning_rate)
        self.training_rewards = []  # 记录训练奖励
        self.reset_buffer()
        self.memory_optimizer = MemoryOptimizer()

    def clear_episode_memory(self):
        """在每个episode结束后强制清理内存"""
        # 1. 清理buffer
        self.reset_buffer()

        # 2. 清理计算图中的中间变量
        for attr in ['_step_indices_last', '_B_last', '_gc_counter']:
            if hasattr(self, attr):
                delattr(self, attr)

        # 3. 强制Python垃圾回收
        import gc
        gc.collect()

        # 4. 清理GPU缓存
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # 5. 清理计算图缓存
        torch.cuda.synchronize()  # 等待所有GPU操作完成

    def reset_buffer(self):
        # 显式删除旧数据以释放内存
        if hasattr(self, 'states'):
            del self.states
        if hasattr(self, 'actions'):
            del self.actions
        if hasattr(self, 'logps'):
            del self.logps
        if hasattr(self, 'rewards'):
            del self.rewards
        if hasattr(self, 'dones'):
            del self.dones
        if hasattr(self, 'values'):
            del self.values
        if hasattr(self, 'dynamic_sizes'):
            del self.dynamic_sizes
        for attr in ['sample_indices', 'feat_values', 'remaining_store', 'remaining_imp_store', 'remaining_nsk_store']:
            if hasattr(self, attr):
                delattr(self, attr)
        # 重新初始化为空列表
        self.states = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.dynamic_sizes = []
        # 紧凑缓冲：不存 512 份完整 state（每份含 512 维 omega），只存 (b_idx, a_i, remaining, ...)，update 时用一份 omega_batch 重建
        self.sample_indices = []
        self.feat_values = []
        self.remaining_store = []
        self.remaining_imp_store = []
        self.remaining_nsk_store = []

    def select(self, s_np: np.ndarray, remaining_bits: int,
               remaining_imp: int = None, remaining_nsk: int = None,
               index_to_action: dict = None) -> Tuple[int, float, float, int]:
        if self.cfg.use_ibp:
            dynamic_size = min(remaining_bits, self.cfg.action_max_per_step)
            # IBP含0比特，(0,0)为合法动作，无需max(1,...)
        else:
            # QAM: 含0比特，动作0..min(remaining,action_max_per_step)
            dynamic_size = min(remaining_bits, self.cfg.action_max_per_step) + 1
        if isinstance(s_np, torch.Tensor):
            s = s_np.detach().to(device).float().unsqueeze(0)
        else:
            s = torch.tensor(s_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            if self.cfg.use_ibp and remaining_imp is not None and remaining_nsk is not None and index_to_action is not None:
                logits = self.actor.net(s).clone()
                # 仅 imp/nsk 两约束（total 可由 imp+nsk 推出）
                for idx in range(logits.shape[1]):
                    if idx not in index_to_action:
                        logits[0, idx] = -1e9
                        continue
                    total_bits, imp_bits = index_to_action[idx]
                    nsk_bits = total_bits - imp_bits
                    # 硬截断：直接禁止超预算动作（与 QAM 一致）
                    if imp_bits > remaining_imp or nsk_bits > remaining_nsk:
                        logits[0, idx] = -1e9
                dist = torch.distributions.Categorical(logits=logits)
            else:
                dist = self.actor(s, dynamic_size)
            a = dist.sample()
            logp = dist.log_prob(a)
            v = self.critic(s)
        if self.cfg.use_ibp:
            action = int(a.item())
        else:
            # QAM: 动作索引k即比特数（0..dynamic_size-1）
            action = int(a.item())
        return action, float(logp.item()), float(v.item()), dynamic_size

    def select_batch(self, states_input: Union[np.ndarray, torch.Tensor], remaining_list: List[int],
                     remaining_imp_list: Optional[List[int]] = None, remaining_nsk_list: Optional[List[int]] = None,
                     index_to_action: Optional[dict] = None) -> Tuple[List[int], List[float], List[float]]:
        """批量 actor/critic 前向。states_input 可为 [N, state_dim] 的 numpy 或已在 device 上的 tensor，减少 CPU↔GPU 传输。
        在评估模式下（actor.eval()），使用确定性策略（argmax）而不是随机采样。"""
        if isinstance(states_input, torch.Tensor):
            s = states_input.detach().to(device).float()
            N = s.shape[0]
        else:
            N = states_input.shape[0]
            s = torch.as_tensor(states_input, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = self.actor.net(s)
            if self.cfg.use_ibp and remaining_imp_list is not None and remaining_nsk_list is not None and index_to_action is not None:
                rem_imp = torch.tensor(remaining_imp_list, dtype=torch.float32, device=device)
                rem_nsk = torch.tensor(remaining_nsk_list, dtype=torch.float32, device=device)
                for idx in range(logits.shape[1]):
                    if idx not in index_to_action:
                        logits[:, idx] = -1e9
                        continue
                    total_bits, imp_bits = index_to_action[idx]
                    nsk_bits = total_bits - imp_bits
                    # 硬截断：直接禁止超预算动作（与 QAM 一致）
                    invalid = (imp_bits > rem_imp) | (nsk_bits > rem_nsk)
                    logits[invalid, idx] = -1e9
                dist = torch.distributions.Categorical(logits=logits)
            else:
                # QAM: 含0比特；动作索引k对应k比特，k > remaining[b] 则非法
                rem_bits_tensor = torch.tensor(remaining_list, dtype=torch.float32, device=device)
                dyn = torch.tensor([min(r, self.cfg.action_max_per_step) + 1 for r in remaining_list], device=device)
                max_dyn = int(dyn.max().item())
                logits = logits[:, :max_dyn]  # QAM动作空间：0到max_dyn-1比特，共max_dyn个动作
                action_bits_grid = torch.arange(0, max_dyn, device=device, dtype=rem_bits_tensor.dtype).unsqueeze(0)
                invalid = action_bits_grid > rem_bits_tensor.unsqueeze(1)  # [N, max_dyn]
                logits = logits.masked_fill(invalid, -1e9)
                dist = torch.distributions.Categorical(logits=logits)
            # 训练和评估默认均使用随机采样策略，与 PPO 训练分布保持一致（是否采样由上层控制 actor.train()/eval() 与随机种子）
            a = dist.sample()
            logp = dist.log_prob(a)
            v = self.critic(s)
        if self.cfg.use_ibp:
            actions = a.cpu().tolist()
        else:
            actions = [int(x) for x in a.cpu().tolist()]  # QAM: 索引k即比特数（0..max_dyn-1）
        logps = logp.cpu().tolist()
        values = v.cpu().tolist()
        return actions, logps, values

    def store(self, s, a, logp, r, v, done, dyn):
        self.states.append(s)
        self.actions.append(a)
        self.logps.append(logp)
        self.rewards.append(r)
        self.values.append(v)
        self.dones.append(done)
        self.dynamic_sizes.append(dyn)

    def store_compact(self, b_idx: int, a_i: float, remaining: int, rem_imp, rem_nsk, a, logp, r, v, done, dyn):
        """紧凑存储：不存完整 state（513 维含 omega），只存 (b_idx, a_i, remaining, ...)，update 时用 omega_batch 重建，大幅省显存。"""
        self.sample_indices.append(b_idx)
        self.feat_values.append(a_i)
        self.remaining_store.append(remaining)
        self.remaining_imp_store.append(rem_imp)
        self.remaining_nsk_store.append(rem_nsk)
        self.states.append(None)  # 占位，使 len(ppo.states) 正确供 step_indices 使用
        self.actions.append(a)
        self.logps.append(logp)
        self.rewards.append(r)
        self.values.append(v)
        self.dones.append(done)
        self.dynamic_sizes.append(dyn)

    def compute_gae(self, last_value: float = 0.0, step_indices=None, batch_size=None):
        """Compute advantage and return. 若提供 step_indices 与 batch_size，则按论文 Eq.(17) 使用 Rt 与 mt=Rt-et。"""
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        adv = np.zeros_like(rewards, dtype=np.float32)
        ret = np.zeros_like(rewards, dtype=np.float32)
        gamma = float(self.cfg.gamma)

        if step_indices is not None and batch_size is not None and batch_size > 0:
            # 论文用 Rt 与 mt=Rt-et（叙述为累计回报）。实践中需使用“从当前步到终点”的 return-to-go：
            # Rt = Σ_{k=t}^{T-1} γ^{k-t} r_k  （每个样本一条长度<=512的轨迹）
            buf_to_bt = {}
            for b in range(batch_size):
                for t in range(len(step_indices[b])):
                    idx = step_indices[b][t]
                    if idx < len(rewards):
                        buf_to_bt[idx] = (b, t)
            for i in range(len(rewards)):
                if i in buf_to_bt:
                    b, t = buf_to_bt[i]
                    Rt = 0.0
                    for k in range(t, len(step_indices[b])):
                        idx_k = step_indices[b][k]
                        if idx_k < len(rewards):
                            Rt += (gamma ** (k - t)) * float(rewards[idx_k])
                    ret[i] = Rt
                    adv[i] = Rt - values[i]
                else:
                    ret[i] = rewards[i]
                    adv[i] = 0.0
            if len(adv) > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            return adv, ret

        last_adv = 0.0
        for t in reversed(range(len(rewards))):
            next_val = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            last_adv = delta + gamma * self.cfg.gae_lambda * (1 - dones[t]) * last_adv
            adv[t] = last_adv
            # value target：R_t = A_t + V(s_t)
            ret[t] = adv[t] + values[t]
        if len(adv) > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def update(self):
        if len(self.states) == 0:
            return
        # 紧凑缓冲：用一份 omega_batch 重建 states，避免 512*B 份完整 state 占满显存
        omega_batch = getattr(self, "omega_batch_for_update", None)
        if omega_batch is not None and len(self.sample_indices) > 0:
            dev = omega_batch.device
            feats_col = torch.tensor(self.feat_values, device=dev, dtype=torch.float32)
            idx = torch.tensor(self.sample_indices, device=dev, dtype=torch.long)
            omega = omega_batch[idx]
            imp_budget = getattr(self, "initial_imp_budget_for_update", None)
            nsk_budget = getattr(self, "initial_nsk_budget_for_update", None)
            pi_u = getattr(self, "pi_for_update", None)
            pu_u = getattr(self, "pu_for_update", None)
            states = build_state(feats_col, omega, self.remaining_store,
                                 self.remaining_imp_store if (imp_budget is not None) else None,
                                 self.remaining_nsk_store if (nsk_budget is not None) else None,
                                 1000, imp_budget, nsk_budget, pi=pi_u, pu=pu_u)
            self.omega_batch_for_update = None
            self.pi_for_update = None
            self.pu_for_update = None
        else:
            states_list = []
            for s in self.states:
                if s is None:
                    continue
                if isinstance(s, torch.Tensor):
                    states_list.append(s.to(device).float())
                elif isinstance(s, np.ndarray):
                    states_list.append(torch.tensor(s, dtype=torch.float32, device=device))
                else:
                    states_list.append(torch.tensor(np.array(s), dtype=torch.float32, device=device))
            if len(states_list) == 0:
                return
            states = torch.stack(states_list)
        # IBP模式存储的是原始索引；QAM模式存储的是1..k，需要减一与分布对齐
        actions = torch.tensor(np.array(self.actions), dtype=torch.long, device=device)
        old_logps = torch.tensor(np.array(self.logps), dtype=torch.float32, device=device)
        dyn_sizes = torch.tensor(np.array(self.dynamic_sizes), dtype=torch.long, device=device)
        with torch.no_grad():
            last_v = self.critic(states[-1:].detach()).item()
        step_indices = getattr(self, "_step_indices_last", None)
        batch_size = getattr(self, "_B_last", None)
        if step_indices is not None and batch_size is not None:
            adv, ret = self.compute_gae(last_v, step_indices=step_indices, batch_size=batch_size)
            self._step_indices_last = None
            self._B_last = None
        else:
            adv, ret = self.compute_gae(last_v)
        adv = torch.tensor(adv, dtype=torch.float32, device=device)
        ret = torch.tensor(ret, dtype=torch.float32, device=device)
        ret_scale = (ret.std().item() + 1e-8) if self.cfg.normalize_value_target else 1.0

        idx = torch.randperm(states.size(0))
        for _ in range(self.cfg.update_epochs):
            start = 0
            while start < states.size(0):
                end = min(start + self.cfg.minibatch, states.size(0))
                b = idx[start:end]
                s = states[b]
                a = actions[b]
                olog = old_logps[b]
                gae = adv[b]
                tg = ret[b]
                dyn = dyn_sizes[b]

                # IBP 和 QAM 都使用批量前向，避免逐样本循环导致的性能瓶颈
                if self.cfg.use_ibp:
                    dist = self.actor(s, None)
                    action_val = a.clamp(0, dist.logits.shape[1] - 1)
                    new_logps = dist.log_prob(action_val)
                    entropy = dist.entropy().mean()
                else:
                    # QAM模式：rollout 中 action 就是“比特数=动作索引”（0..max_dyn-1，含0比特）
                    max_dyn = int(dyn.max().item())
                    dist = self.actor(s, max_dyn)
                    action_val = a.clamp(0, max_dyn - 1)
                    invalid_action_mask = action_val >= dyn  # dyn 是每个样本的可用动作数（含0比特）
                    log_probs = dist.log_prob(action_val)
                    log_probs[invalid_action_mask] = -1e9
                    new_logps = log_probs
                    dyn_expanded = dyn.unsqueeze(1)
                    action_indices = torch.arange(max_dyn, device=s.device).unsqueeze(0)
                    invalid_mask = action_indices >= dyn_expanded
                    entropy_logits = dist.logits.clone()
                    entropy_logits[invalid_mask] = -1e9
                    entropy_dist = torch.distributions.Categorical(logits=entropy_logits)
                    entropy = entropy_dist.entropy().mean()

                ratio = torch.exp(new_logps - olog)
                surr1 = ratio * gae
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * gae
                policy_loss = -torch.min(surr1, surr2).mean()

                v = self.critic(s)
                value_loss = F.mse_loss(v, tg) / ret_scale

                self.opt_cri.zero_grad()
                (self.cfg.value_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.opt_cri.step()

                self.opt_act.zero_grad()
                actor_total = policy_loss - self.cfg.entropy_coef * entropy  # 减去熵鼓励探索
                actor_total.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.opt_act.step()

                start = end

        # 清理内存：释放中间变量和GPU缓存
        del states, actions, old_logps, dyn_sizes, adv, ret, idx, s, a, olog, gae, tg, dyn
        if device.type == "cuda":
            torch.cuda.empty_cache()  # 清理GPU未使用的缓存

        self.reset_buffer()

        # Python垃圾回收（每N次update执行一次，避免过于频繁）
        if not hasattr(self, "_gc_counter"):
            self._gc_counter = 0
        self._gc_counter += 1
        if self._gc_counter % 50 == 0:  # 每50次update执行一次GC，平衡内存与速度
            gc.collect()


# ==================== Training Phases ====================
def pretrain_encoder_decoder(encoder: nn.Module, decoder: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                             epochs_head: int = 100, epochs_finetune: int = 100, snr_db: float = 1000.0,
                             save_paths: Tuple[str, str] = ("best_encoder.pth", "best_decoder.pth"),
                             joint_only: bool = True, patience: int = 15):
    """
    语义编解码器训练。
    - joint_only=True（默认）：联合训练 encoder+decoder，端到端优化分类损失（经 AWGN 信道）。
    - joint_only=False：两阶段——Phase1 只训 decoder（encoder 冻结），Phase2 再联合微调。
    - val_loader：必须在测试集/验证集上评估以选最优模型；若用训练集会导致过拟合、准确率随训练下降。
    - patience：验证集准确率连续 patience 个 epoch 不提升则早停。
    """
    awgn = AWGN().to(device)
    enc_path, dec_path = save_paths
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    if not joint_only:
        for p in encoder.parameters():
            p.requires_grad = False
        optim_dec = optim.SGD(decoder.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
        for _ in tqdm(range(epochs_head), desc="Pretrain Phase1 (decoder only)"):
            encoder.eval()
            decoder.train()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.no_grad():
                    feats = encoder(imgs)
                    feats_noisy = awgn(feats, snr_db)
                logits = decoder(feats_noisy)
                loss = ce(logits, labels)
                optim_dec.zero_grad()
                loss.backward()
                optim_dec.step()
        for p in encoder.parameters():
            p.requires_grad = True

    optim_all = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3, momentum=0.9,
                          weight_decay=1e-4)
    # CosineAnnealing 比 StepLR 更温和，避免后期 LR 骤降至 1e-9 导致数值不稳、准确率下降
    n_epochs = epochs_finetune if not joint_only else (epochs_head + epochs_finetune)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_all, T_max=n_epochs, eta_min=1e-6)
    best_acc = 0.0
    no_improve_count = 0
    desc = "Joint train (encoder+decoder)" if joint_only else "Pretrain Phase2 (finetune all)"
    for ep in tqdm(range(n_epochs), desc=desc):
        encoder.train()
        decoder.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = encoder(imgs)
            feats_noisy = awgn(feats, snr_db)
            logits = decoder(feats_noisy)
            loss = ce(logits, labels)
            optim_all.zero_grad()
            loss.backward()
            optim_all.step()
        scheduler.step()

        encoder.eval()
        decoder.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats = encoder(imgs)
                logits = decoder(feats)
                pred = logits.argmax(dim=1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        acc = 100.0 * correct / max(1, total)
        if acc > best_acc:
            best_acc = acc
            no_improve_count = 0
            torch.save(encoder.state_dict(), enc_path)
            torch.save(decoder.state_dict(), dec_path)
        else:
            no_improve_count += 1
        if no_improve_count >= patience:
            tqdm.write(f"早停: 验证准确率 {patience} epoch 未提升，最佳 {best_acc:.2f}%")
            break


def _compute_pi_pu_for_state(snr_db: float, alpha: float, imp_bits: int = 2, nsk_bits: int = 4,
                             suffix_bits: int = 8) -> tuple:
    """根据 SNR 和 alpha 计算重要比特/非重要比特的误比特率 Pi、Pu，用于状态输入。"""
    if not _SER_AVAILABLE:
        return 0.05, 0.10
    try:
        if not hasattr(ser, "nsk_map") or not ser.nsk_map:
            ser.generate_nsk_map(1)
        _, p_prefix, p_suffix = ser.compute_theoretical_nsk_ser(snr_db, float(alpha), suffix_bits)
        p_prefix = max(0.0, min(1.0, p_prefix))
        p_suffix = max(0.0, min(1.0, p_suffix))
        Pi = 1.0 - (1.0 - p_prefix) ** (1.0 / max(1, imp_bits))
        Pu = 1.0 - (1.0 - p_suffix) ** (1.0 / max(1, nsk_bits))
        return float(Pi), float(Pu)
    except Exception:
        return 0.05, 0.10


def build_state(feats_col: torch.Tensor, omega: torch.Tensor,
                remaining_list: List[int], remaining_imp_list: List[int] = None,
                remaining_nsk_list: List[int] = None,
                total_bits: int = 1000, initial_imp_budget: int = None,
                initial_nsk_budget: int = None,
                pi: float = None, pu: float = None) -> torch.Tensor:
    """在GPU上批量构建状态 [N, state_dim]。IBP 模式下可传入 pi, pu（误比特率）以扩展状态维度。"""
    with torch.no_grad():
        N = feats_col.shape[0]
        if omega.dim() == 2 and omega.shape[0] == N:
            omega_exp = omega.float()
        else:
            omega_exp = omega.float().unsqueeze(0).expand(N, -1)

        total_bits_f = float(max(1, total_bits))
        rem_ratio = torch.tensor([r / total_bits_f for r in remaining_list], device=omega.device,
                                 dtype=torch.float32).unsqueeze(1)
        init_imp = float(max(1, initial_imp_budget)) if initial_imp_budget is not None else total_bits_f
        init_nsk = float(max(1, initial_nsk_budget)) if initial_nsk_budget is not None else total_bits_f
        if remaining_imp_list is not None and remaining_nsk_list is not None:
            rem_imp_ratio = torch.tensor([r / init_imp for r in remaining_imp_list], device=omega.device,
                                         dtype=torch.float32).unsqueeze(1)
            rem_nsk_ratio = torch.tensor([r / init_nsk for r in remaining_nsk_list], device=omega.device,
                                         dtype=torch.float32).unsqueeze(1)
        else:
            rem_imp_ratio = rem_ratio
            rem_nsk_ratio = rem_ratio

        parts = [
            feats_col.unsqueeze(1),
            omega_exp,
            rem_ratio,
            rem_imp_ratio,
            rem_nsk_ratio,
        ]
        if pi is not None and pu is not None:
            pi_t = torch.full((N, 1), pi, device=omega.device, dtype=torch.float32)
            pu_t = torch.full((N, 1), pu, device=omega.device, dtype=torch.float32)
            parts.extend([pi_t, pu_t])
        state = torch.cat(parts, dim=1)
    return state


def weighted_mse(x: torch.Tensor, y: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """
    论文 Eq.(10): d(A, A'; b, ω, ρ) = Σ_{i=1}^C ωi |ai - a'i|²，对**所有**语义 i 求和。

    - 已分配比特的语义：a'i = 信道重建；未分配比特的语义：a'i = 0（未传输）。
    - 因此 d 始终对全部 C 维求和，不会因「只对已分配求和」而在无分配时得 0。
    """
    return (omega * (x - y) ** 2).sum()


def _paper_mse_per_action(imp_bits: int, nsk_bits: int, Pi: float, Pu: float) -> float:
    """
    统一 MSE 公式（满足 A=0 或 B=0 的边界）：
    E(MSE) = 1/(3·4^n) + (4·Pi/3)·(1 - 1/4^A) + (4·Pu/3)·(1/4^A - 1/4^n)
    A=重要比特数, B=非重要比特数, n=A+B; Pi=重要比特BER, Pu=非重要比特BER
    三项分别为：量化底噪、重要比特引起的误差、非重要比特引起的误差
    """
    A, B = imp_bits, nsk_bits
    n = A + B
    if n == 0:
        return 1.0 / 3.0
    four_n = 4.0 ** n
    four_A = 4.0 ** A
    e_quan = 1.0 / (3.0 * four_n)
    e_imp = (4.0 * Pi / 3.0) * (1.0 - 1.0 / four_A)
    e_unimp = (4.0 * Pu / 3.0) * (1.0 / four_A - 1.0 / four_n)
    return e_quan + e_imp + e_unimp


def _debug_alpha_bit_distribution(bit_allocation: List[List[Tuple]]) -> None:
    """调试用：打印 bit_allocation 中 (imp,nsk) 的分布，便于理解最优 alpha 为何在边界。"""
    from collections import Counter
    pairs = []
    for b in bit_allocation:
        for alloc in b:
            if len(alloc) >= 4 and alloc[1] > 0:
                pairs.append((alloc[2], alloc[3]))
    if not pairs:
        return
    cnt = Counter(pairs)
    top = cnt.most_common(5)
    print(f"  [Alpha debug] (imp,nsk) dist: {top}")


def _qam_partial_L_d_batched(items, quantizer, qam_mod, decoder, snr_db, device=None):
    """批量计算每步 L、d：items = [(feats, partial, omega, label), ...]，返回 list of (L, d)。device 可选，用于双 GPU 时指定设备。"""
    with torch.no_grad():
        if not items:
            return []
        dev = device if device is not None else items[0][0].device
        feats = torch.stack([x[0] for x in items])
        omega = torch.stack([x[2] for x in items])
        labels = torch.stack([x[3].squeeze() for x in items])
        list_partials = [x[1] for x in items]
        # 区分空部分分配与非空
        empty_idx = [i for i, p in enumerate(list_partials) if not p]
        non_empty_idx = [i for i, p in enumerate(list_partials) if p]
        out_L = [0.0] * len(items)
        out_d = [0.0] * len(items)
        if empty_idx:
            rec_empty = torch.zeros(len(empty_idx), feats.shape[1], device=dev, dtype=feats.dtype)
            with torch.no_grad():
                logits = decoder(rec_empty)
                pred = logits.argmax(dim=1)
                for j, i in enumerate(empty_idx):
                    out_L[i] = 1.0 if pred[j].item() == labels[i].item() else 0.0
                    out_d[i] = (omega[i] * (feats[i]) ** 2).sum().item()
        if not non_empty_idx:
            return [(out_L[i], out_d[i]) for i in range(len(items))]
        # 非空：为每个样本构建比特并 pad 到同一长度（6 的倍数）
        bits_list = []
        num_bits_list = []
        partials_sub = [list_partials[i] for i in non_empty_idx]
        feats_sub = feats[non_empty_idx]
        for n, partial in enumerate(partials_sub):
            all_bits = []
            for (feat_idx, bits) in partial:
                q = quantizer.quantize_feature(feats_sub[n, feat_idx], bits)
                M = 2 ** bits
                int_val = ((q + 1.0) * M / 2).clamp(0, M - 1).long().item()
                for bi in range(bits):
                    all_bits.append(int((int_val >> bi) & 1))
            num_bits_list.append(len(all_bits))
            bits_list.append(all_bits)
        max_bits = max(num_bits_list)
        pad_to = ((max_bits + 5) // 6) * 6
        padded = []
        for bl in bits_list:
            p = bl + [0] * (pad_to - len(bl))
            padded.append(p)
        bits_tensor = torch.tensor(padded, device=dev, dtype=torch.float32)
        symbols = qam_mod.modulate(bits_tensor)
        noise_power = _noise_power_from_symbols(symbols, snr_db)
        noise_real = torch.randn_like(symbols.real) * math.sqrt(noise_power / 2)
        noise_imag = torch.randn_like(symbols.imag) * math.sqrt(noise_power / 2)
        noisy = symbols + torch.complex(noise_real, noise_imag)
        demod = qam_mod.demodulate(noisy)
        rec_sub = torch.zeros(len(non_empty_idx), feats.shape[1], device=dev, dtype=feats.dtype)
        for n in range(len(non_empty_idx)):
            partial = partials_sub[n]
            bit_idx = 0
            for (feat_idx, bits) in partial:
                if bit_idx + bits <= demod.shape[1]:
                    fb = demod[n, bit_idx:bit_idx + bits]
                    int_val = 0
                    for j in range(min(bits, len(fb))):
                        if fb[j] > 0.5:
                            int_val += (1 << j)
                    rec_sub[n, feat_idx] = dequantize_from_int(int_val, bits, device_=dev).squeeze()
                    bit_idx += bits
        with torch.no_grad():
            logits = decoder(rec_sub)
            pred = logits.argmax(dim=1)
            for j, i in enumerate(non_empty_idx):
                out_L[i] = 1.0 if pred[j].item() == labels[i].item() else 0.0
                out_d[i] = weighted_mse(feats[i], rec_sub[j], omega[i]).item()
    return [(out_L[i], out_d[i]) for i in range(len(items))]


def _ibp_partial_L_d_batched(items, ibp_quantizer, ibp_mod, decoder, snr_db, device=None):
    """批量计算每步 L、d。device 可选，用于双 GPU。"""
    if not items:
        return []
    dev = device if device is not None else items[0][0].device
    feats = torch.stack([x[0] for x in items])
    omega = torch.stack([x[2] for x in items])
    labels = torch.stack([x[3].squeeze() for x in items])
    list_partials = [x[1] for x in items]
    empty_idx = [i for i, p in enumerate(list_partials) if not p]
    non_empty_idx = [i for i, p in enumerate(list_partials) if p]
    out_L = [0.0] * len(items)
    out_d = [0.0] * len(items)
    if empty_idx:
        rec_empty = torch.zeros(len(empty_idx), feats.shape[1], device=dev, dtype=feats.dtype)
        with torch.no_grad():
            logits = decoder(rec_empty)
            pred = logits.argmax(dim=1)
            for j, i in enumerate(empty_idx):
                out_L[i] = 1.0 if pred[j].item() == labels[i].item() else 0.0
                out_d[i] = (omega[i] * (feats[i]) ** 2).sum().item()
    if not non_empty_idx:
        return [(out_L[i], out_d[i]) for i in range(len(items))]
    bits_list = []
    partials_sub = [list_partials[i] for i in non_empty_idx]
    feats_sub = feats[non_empty_idx]
    for n, partial in enumerate(partials_sub):
        imp_stream, nsk_stream = [], []
        for alloc in partial:
            feat_idx, total_bits, imp_bits, nsk_bits = alloc[0], alloc[1], alloc[2], alloc[3]
            quantized, _, int_val_opt = ibp_quantizer.quantize_with_ibp(feats_sub[n, feat_idx], total_bits, imp_bits)
            M_total = 2 ** total_bits
            int_val = int_val_opt if int_val_opt is not None else ((quantized + 1.0) * M_total / 2).clamp(0,
                                                                                                          M_total - 1).long().item()
            imp_val = int_val >> (total_bits - imp_bits) if imp_bits > 0 else 0
            nsk_val = int_val & ((1 << (total_bits - imp_bits)) - 1) if nsk_bits > 0 else 0
            for jj in range(imp_bits):
                imp_stream.append(int((imp_val >> (imp_bits - 1 - jj)) & 1))
            for jj in range(nsk_bits):
                nsk_stream.append(int((nsk_val >> (nsk_bits - 1 - jj)) & 1))
        packed = pack_ibp_dual_stream(imp_stream, nsk_stream, device=dev)
        bits_list.append(packed.squeeze(0))
    max_len = max(t.shape[0] for t in bits_list)
    pad_to = ((max_len + 5) // 6) * 6
    padded = []
    for t in bits_list:
        if t.shape[0] < pad_to:
            t = torch.cat([t, torch.zeros(pad_to - t.shape[0], device=dev, dtype=torch.float32)])
        padded.append(t)
    bits_tensor = torch.stack(padded)
    symbols = ibp_mod.modulate(bits_tensor, snr_db=snr_db)
    demod = ibp_mod.demodulate(symbols)
    rec_sub = torch.zeros(len(non_empty_idx), feats.shape[1], device=dev, dtype=feats.dtype)
    for n in range(len(non_empty_idx)):
        partial = partials_sub[n]
        unpacked = unpack_ibp_dual_stream(demod, partial, batch_idx=n, device=dev)
        for feat_idx, int_val, total_bits in unpacked:
            rec_sub[n, feat_idx] = dequantize_from_int(int_val, total_bits, device_=dev).squeeze()
    with torch.no_grad():
        logits = decoder(rec_sub)
        pred = logits.argmax(dim=1)
        for j, i in enumerate(non_empty_idx):
            out_L[i] = 1.0 if pred[j].item() == labels[i].item() else 0.0
            out_d[i] = weighted_mse(feats[i], rec_sub[j], omega[i]).item()
    return [(out_L[i], out_d[i]) for i in range(len(items))]


# 批量部分奖励时双 GPU 拆分 decoder 的最小样本数，低于则单卡
_MIN_BATCH_DUAL_GPU = 32


def _qam_partial_rewards_batched_all(feats_batch, B, all_bits_list, bit_allocation, step_indices, step_legal_count,
                                     omega_batch, labels, ppo, qam_mod, decoder, current_snr, decoder_cuda1=None):
    """将 512 步部分重建+解码合并为一次批量计算：一次调制/信道/解调 + 一次 decoder(N,512)，充分利用 GPU。
    若 decoder_cuda1 非空且 N>=_MIN_BATCH_DUAL_GPU，则 rec 拆为两半在双 GPU 上并行解码。直接写入 ppo.rewards。"""
    with torch.inference_mode():
        indices_to_update = [(b, k) for b in range(B) for k in range(len(step_indices[b]))]
        if not indices_to_update:
            return
        N = len(indices_to_update)
        dev = feats_batch.device
        feat_dim = feats_batch.shape[1]
        # 每步 (b,k) 的累积比特数
        cum_bits = []
        for (b, k) in indices_to_update:
            n_legal = step_legal_count[b][k]
            c = sum(bit_allocation[b][j][1] for j in range(n_legal))
            cum_bits.append(c)
        max_bits_val = max(cum_bits)
        rec = torch.zeros(N, feat_dim, device=dev, dtype=feats_batch.dtype)
        if max_bits_val > 0:
            pad_to = ((max_bits_val + 5) // 6) * 6
            arr = np.zeros((N, pad_to), dtype=np.float32)
            all_bits_arr = [np.array(all_bits_list[b], dtype=np.float32) for b in range(B)]
            for n, (b, k) in enumerate(indices_to_update):
                c = cum_bits[n]
                if c > 0:
                    arr[n, :c] = all_bits_arr[b][:c]
            bits_tensor = torch.from_numpy(arr).to(dev)
            symbols = qam_mod.modulate(bits_tensor)
            noise_power = _noise_power_from_symbols(symbols, current_snr)
            noise_real = torch.randn_like(symbols.real, device=dev) * math.sqrt(noise_power / 2)
            noise_imag = torch.randn_like(symbols.imag, device=dev) * math.sqrt(noise_power / 2)
            noisy = symbols + torch.complex(noise_real, noise_imag)
            demod = qam_mod.demodulate(noisy)
            demod_np = demod.cpu().numpy()
            for n in range(N):
                b, k = indices_to_update[n]
                n_legal = step_legal_count[b][k]
                partial = bit_allocation[b][:n_legal]
                bit_idx = 0
                for (feat_idx, bits) in partial:
                    if bit_idx + bits <= demod.shape[1]:
                        fb = demod_np[n, bit_idx:bit_idx + bits]
                        int_val = int(np.dot((fb > 0.5).astype(np.uint8), 2 ** np.arange(len(fb), dtype=np.uint64)))
                        rec[n, feat_idx] = dequantize_from_int(int_val, bits, device_=dev).squeeze()
                        bit_idx += bits
        use_dual = (decoder_cuda1 is not None and N >= _MIN_BATCH_DUAL_GPU and rec.is_cuda)
        if use_dual:
            mid = N // 2
            rec0, rec1 = rec[:mid], rec[mid:].to('cuda:1', non_blocking=True)

            def run0():
                with torch.amp.autocast('cuda'):
                    return decoder(rec0).float()

            def run1():
                with torch.amp.autocast('cuda'):
                    return decoder_cuda1(rec1).float()

            with ThreadPoolExecutor(max_workers=2) as ex:
                f0 = ex.submit(run0)
                f1 = ex.submit(run1)
                logits0 = f0.result()
                logits1 = f1.result()
            logits = torch.cat([logits0, logits1.to(dev)], dim=0)
        elif rec.is_cuda:
            with torch.amp.autocast('cuda'):
                logits = decoder(rec)
            logits = logits.float()
        else:
            logits = decoder(rec)
        pred = logits.argmax(dim=1)
        beta = ppo.cfg.beta
        use_diff = getattr(ppo.cfg, "use_difference_reward", True)
        b_indices = [indices_to_update[n][0] for n in range(N)]
        feats_by_n = feats_batch[b_indices]
        omega_by_n = omega_batch[b_indices]
        labels_b = labels[b_indices].squeeze() if labels.numel() > 1 else labels.expand(N)
        dk_all = (omega_by_n * (feats_by_n - rec) ** 2).sum(dim=1)
        Lk_all = (pred == labels_b).float()
        dk_np = dk_all.cpu().numpy()
        Lk_np = Lk_all.cpu().numpy()
        if use_diff:
            d_empty = (omega_batch * (feats_batch ** 2)).sum(dim=1).cpu().numpy()
            prev_U = {b: -beta * d_empty[b] for b in range(B)}
        for n in range(N):
            b, k = indices_to_update[n]
            idx = step_indices[b][k]
            Lk = float(Lk_np[n])
            dk = float(dk_np[n])
            if use_diff:
                U_curr = Lk - beta * dk
                r_k = U_curr - prev_U[b]
                prev_U[b] = U_curr
                is_last = (k == len(step_indices[b]) - 1)
                ppo.rewards[idx] = r_k + (ppo.cfg.L0 if is_last else 0.0)
            else:
                is_last_step = (k == len(step_indices[b]) - 1)
                ppo.rewards[idx] = (ppo.cfg.L0 if is_last_step else 0.0) + Lk - beta * dk


def _run_qam_batched_round(feats_batch, B, labels, order_inds_batch, omega_batch, ppo, quantizer, qam_mod, decoder,
                           current_snr, decoder_cuda1=None):
    """QAM 批量前向。order_inds_batch[b] 为样本 b 的特征步进顺序（固定 0..D-1），omega_batch [B, D] 为每样本重要性。
    若 decoder_cuda1 非空则每轮同步参数并传入批量部分奖励以双 GPU 并行解码。"""
    ppo.omega_batch_for_update = omega_batch
    ppo.initial_imp_budget_for_update = None
    ppo.initial_nsk_budget_for_update = None
    total_budget = 1000
    remaining = [total_budget] * B
    bit_allocation = [[] for _ in range(B)]
    all_bits_list = [[] for _ in range(B)]
    total_used_bits = [0] * B
    step_indices = [[] for _ in range(B)]
    step_feat_indices = [[] for _ in range(B)]
    step_legal_count = [[] for _ in range(B)]  # 每步后当前合法分配数，用于论文 b(i) 部分分配
    omega_dev = omega_batch.to(feats_batch.device)
    for t in range(512):
        active = [b for b in range(B) if remaining[b] > 0]
        if not active:
            break
        feats_col = torch.stack([feats_batch[b, order_inds_batch[b][t]] for b in active])
        omega_active = omega_dev[active]
        states_tensor = build_state(
            feats_col, omega_active,
            [remaining[b] for b in active], None, None,  # QAM没有imp/nsk预算
            1000, None, None)
        actions, logps, values = ppo.select_batch(states_tensor, [remaining[b] for b in active], None, None, None)
        to_quant_j, to_quant_b, to_quant_i, to_quant_bits = [], [], [], []
        for j, b in enumerate(active):
            i = order_inds_batch[b][t]
            action_bits = actions[j].item() if isinstance(actions[j], torch.Tensor) else actions[j]
            dyn_size = min(remaining[b], ppo.cfg.action_max_per_step) + 1  # QAM: 含0比特，动作数=0..min(...)
            if action_bits < 0 or action_bits > remaining[b]:
                feat_val_tensor = feats_batch[b, i]
                omega_i_tensor = omega_batch[b, i]
                illegal_penalty = -(omega_i_tensor * feat_val_tensor.abs()).item()
                ppo.store_compact(b, feats_batch[b, i].item(), remaining[b], None, None,
                                  action_bits, logps[j], illegal_penalty, values[j], False, dyn_size)
                step_indices[b].append(len(ppo.states) - 1)
                step_feat_indices[b].append(i)
                step_legal_count[b].append(len(bit_allocation[b]))
                continue
            elif action_bits == 0:
                # 选择 0-bit：若仍有预算，则施加惩罚，避免“有预算也不发”
                zb = float(getattr(ppo.cfg, "zero_bit_penalty", 0.0))
                zero_pen = -zb * (remaining[b] / float(total_budget)) if (zb > 0 and remaining[b] > 0) else 0.0
                ppo.store_compact(b, feats_batch[b, i].item(), remaining[b], None, None,
                                  0, logps[j], zero_pen, values[j], False, dyn_size)
                step_indices[b].append(len(ppo.states) - 1)
                step_feat_indices[b].append(i)
                step_legal_count[b].append(len(bit_allocation[b]))
                continue
            else:
                action_bits = min(action_bits, remaining[b], ppo.cfg.action_max_per_step)
                to_quant_j.append(j)
                to_quant_b.append(b)
                to_quant_i.append(i)
                to_quant_bits.append(action_bits)
        if to_quant_j:
            feats_to_q = torch.stack([feats_batch[to_quant_b[k], to_quant_i[k]] for k in range(len(to_quant_j))])
            int_vals = quantizer.quantize_batch_multi_bits(feats_to_q, to_quant_bits)
            for k in range(len(to_quant_j)):
                b, i = to_quant_b[k], to_quant_i[k]
                action_bits = to_quant_bits[k]
                int_val = int(int_vals[k].item())
                bits_arr = ((int_val >> np.arange(action_bits, dtype=np.int64)) & 1).astype(np.float32)
                all_bits_list[b].extend(bits_arr)
                remaining[b] -= action_bits
                total_used_bits[b] += action_bits
                dyn_size = min(remaining[b] + action_bits, ppo.cfg.action_max_per_step) + 1  # remaining[b] 已更新，回退到“动作前”的 remaining
                ppo.store_compact(b, feats_batch[b, i].item(), remaining[b], None, None,
                                  action_bits, logps[to_quant_j[k]], 0.0, values[to_quant_j[k]], False,
                                  dyn_size)
                step_indices[b].append(len(ppo.states) - 1)
                step_feat_indices[b].append(i)
                bit_allocation[b].append((i, action_bits))
                step_legal_count[b].append(len(bit_allocation[b]))
    # 终点剩余比特惩罚：鼓励用满预算（加到每图最后一步 reward 上）
    lb = float(getattr(ppo.cfg, "leftover_bit_penalty", 0.0))
    if lb > 0:
        for b in range(B):
            if remaining[b] > 0 and step_indices[b]:
                idx_last = step_indices[b][-1]
                ppo.rewards[idx_last] = float(ppo.rewards[idx_last]) - lb * (remaining[b] / float(total_budget))
    # 批量处理最终重建以提高GPU利用率（所有样本一起解码）
    list_final_rewards, list_task_perf, list_dist = [], [], []
    rec_features_batch = torch.zeros_like(feats_batch)  # [B, 512]

    # 批量调制和解调（向量化 pack，与 IBP pack_ibp_dual_stream_batch 一致）
    non_empty_mask = torch.tensor([len(all_bits_list[b]) > 0 for b in range(B)], device=device)
    if non_empty_mask.any():
        bits_batch = pack_qam_bits_batch(all_bits_list, device=device)  # [B, pad_to]
        symbols_batch = qam_mod.modulate(bits_batch)
        noise_power = _noise_power_from_symbols(symbols_batch, current_snr)
        noise_real = torch.randn_like(symbols_batch.real) * math.sqrt(noise_power / 2)
        noise_imag = torch.randn_like(symbols_batch.imag) * math.sqrt(noise_power / 2)
        noisy_symbols_batch = symbols_batch + torch.complex(noise_real, noise_imag)
        demod_bits_batch = qam_mod.demodulate(noisy_symbols_batch)  # [B, pad_to]

        # 批量重建特征（一次 cpu().numpy() 避免每分配一次 GPU 同步）
        demod_np = demod_bits_batch.cpu().numpy()
        for b in range(B):
            if len(all_bits_list[b]) > 0:
                bit_idx = 0
                for alloc in bit_allocation[b]:
                    feat_idx = alloc[0]
                    bits = alloc[1]
                    if bit_idx + bits <= demod_bits_batch.shape[1]:
                        fb = demod_np[b, bit_idx:bit_idx + bits]
                        int_val = int(np.dot((fb > 0.5).astype(np.uint8), 2 ** np.arange(len(fb), dtype=np.uint64)))
                        rec_features_batch[b, feat_idx] = dequantize_from_int(int_val, bits, device_=device).squeeze()
                        bit_idx += bits
                    else:
                        break
            # 未分配比特的样本保持 rec 为 0，不填原特征
    else:
        pass  # 无任何样本分配比特时，保持 rec 全 0

    # 批量解码（所有样本一起）；GPU 上用 AMP 加速 decoder 前向
    with torch.no_grad():
        if rec_features_batch.is_cuda:
            with torch.amp.autocast('cuda'):
                logits_batch = decoder(rec_features_batch)  # [B, num_classes]
            logits_batch = logits_batch.float()
        else:
            logits_batch = decoder(rec_features_batch)  # [B, num_classes]
        pred_batch = logits_batch.argmax(dim=1)  # [B]
        task_perf_batch = (pred_batch == labels.squeeze()).float()  # [B]

        # 批量计算distortion（向量化操作）
        # weighted_mse对每个样本: sum(omega * (feats - rec)^2)
        diff = feats_batch - rec_features_batch  # [B, 512]
        dist_batch = (omega_batch * (diff ** 2)).sum(dim=1)  # [B]

        # 批量计算reward
        final_reward_batch = ppo.cfg.L0 + task_perf_batch - ppo.cfg.beta * dist_batch

        list_final_rewards = final_reward_batch.cpu().tolist()
        list_task_perf = task_perf_batch.cpu().tolist()
        list_dist = dist_batch.cpu().tolist()

    # 论文每步奖励 r_i = L0 + L(b(i)) − β·d(b(i))。512 步部分重建+解码一次性批量计算（一次调制/信道/解调 + 一次 decoder），提高 GPU 利用率
    max_steps = max(len(step_indices[b]) for b in range(B)) if B else 0
    if max_steps > 0:
        if decoder_cuda1 is not None:
            decoder_cuda1.load_state_dict(decoder.state_dict())
        _qam_partial_rewards_batched_all(
            feats_batch, B, all_bits_list, bit_allocation, step_indices, step_legal_count,
            omega_batch, labels, ppo, qam_mod, decoder, current_snr, decoder_cuda1=decoder_cuda1,
        )

    # 折扣回报：Eq.(17) 用于 PPO 价值目标与 advantage；无折扣回报：Σ r_t 用于绘图/进度条（与 Fig.7 纵轴 500~3500 一致）
    gamma = getattr(ppo.cfg, "gamma", 0.99)
    list_discounted_returns = []
    list_undiscounted_returns = []
    for b in range(B):
        if not step_indices[b]:
            list_discounted_returns.append(0.0)
            list_undiscounted_returns.append(0.0)
            continue
        G_disc = sum(
            (gamma ** k) * ppo.rewards[step_indices[b][k]]
            for k in range(len(step_indices[b]))
            if step_indices[b][k] < len(ppo.rewards)
        )
        G_undisc = sum(
            ppo.rewards[step_indices[b][k]]
            for k in range(len(step_indices[b]))
            if step_indices[b][k] < len(ppo.rewards)
        )
        list_discounted_returns.append(float(G_disc))
        list_undiscounted_returns.append(float(G_undisc))
    ppo._step_indices_last = step_indices
    ppo._B_last = B
    return list_final_rewards, list_task_perf, list_dist, list_discounted_returns, list_undiscounted_returns


def _ibp_partial_rewards_batched_all(feats_batch, B, imp_stream_list, nsk_stream_list, bit_allocation, step_indices,
                                     step_legal_count,
                                     step_imp_bonuses, omega_batch, labels, ppo, ibp_quantizer, ibp_mod, decoder,
                                     current_snr, decoder_cuda1=None):
    """将 512 步部分重建+解码合并为一次批量计算：一次 IBP 调制(含噪声)/解调 + 一次 decoder(N,512)。
    双流格式：每符号 2 重要 + 4 非重要。直接切片预计算的 imp/nsk 流，避免重复量化。
    奖励：use_difference_reward 时用差分奖励 r_k = (Lk-β·dk) - (L_{k-1}-β·d_{k-1})，避免绝对效用负分累积。"""
    with torch.inference_mode():
        indices_to_update = [(b, k) for b in range(B) for k in range(len(step_indices[b]))]
        if not indices_to_update:
            return
        N = len(indices_to_update)
        dev = feats_batch.device
        feat_dim = feats_batch.shape[1]
        rec = torch.zeros(N, feat_dim, device=dev, dtype=feats_batch.dtype)
        bits_list = []
        for n, (b, k) in enumerate(indices_to_update):
            n_legal = step_legal_count[b][k]
            partial = bit_allocation[b][:n_legal]
            if not partial:
                bits_list.append(torch.zeros(6, device=dev, dtype=torch.float32))
                continue
            cum_imp = sum(alloc[2] for alloc in partial)
            cum_nsk = sum(alloc[3] for alloc in partial)
            imp_partial = imp_stream_list[b][:cum_imp]
            nsk_partial = nsk_stream_list[b][:cum_nsk]
            packed = pack_ibp_dual_stream(imp_partial, nsk_partial, device=dev)
            bits_list.append(packed.squeeze(0))
        if bits_list:
            max_len = max(t.shape[0] for t in bits_list)
            pad_to = ((max_len + 5) // 6) * 6
            padded = []
            for t in bits_list:
                if t.shape[0] < pad_to:
                    t = torch.cat([t, torch.zeros(pad_to - t.shape[0], device=dev, dtype=torch.float32)])
                padded.append(t)
            bits_tensor = torch.stack(padded)
            symbols = ibp_mod.modulate(bits_tensor, snr_db=current_snr)
            demod = ibp_mod.demodulate(symbols)
            for n in range(N):
                b, k = indices_to_update[n]
                n_legal = step_legal_count[b][k]
                partial = bit_allocation[b][:n_legal]
                if partial:
                    unpacked = unpack_ibp_dual_stream(demod, partial, batch_idx=n, device=dev)
                    for feat_idx, int_val, total_bits in unpacked:
                        rec[n, feat_idx] = dequantize_from_int(int_val, total_bits, device_=dev).squeeze()
        use_dual = (decoder_cuda1 is not None and N >= _MIN_BATCH_DUAL_GPU and rec.is_cuda)
        if use_dual:
            mid = N // 2
            rec0, rec1 = rec[:mid], rec[mid:].to('cuda:1', non_blocking=True)

            def run0():
                with torch.amp.autocast('cuda'):
                    return decoder(rec0).float()

            def run1():
                with torch.amp.autocast('cuda'):
                    return decoder_cuda1(rec1).float()

            with ThreadPoolExecutor(max_workers=2) as ex:
                f0 = ex.submit(run0)
                f1 = ex.submit(run1)
                logits0 = f0.result()
                logits1 = f1.result()
            logits = torch.cat([logits0, logits1.to(dev)], dim=0)
        elif rec.is_cuda:
            with torch.amp.autocast('cuda'):
                logits = decoder(rec)
            logits = logits.float()
        else:
            logits = decoder(rec)
        pred = logits.argmax(dim=1)

        beta = ppo.cfg.beta
        use_diff = getattr(ppo.cfg, "use_difference_reward", True)
        if use_diff:
            d_empty = (omega_batch * (feats_batch ** 2)).sum(dim=1).cpu().numpy()
            prev_U = {b: -beta * d_empty[b] for b in range(B)}
        for n in range(N):
            b, k = indices_to_update[n]
            idx = step_indices[b][k]
            Lk = 1.0 if pred[n].item() == labels[b].item() else 0.0
            dk = (omega_batch[b] * (feats_batch[b] - rec[n]) ** 2).sum().item()
            step_bonus = step_imp_bonuses[b][k] if k < len(step_imp_bonuses[b]) else 0.0
            if use_diff:
                U_curr = Lk - beta * dk
                r_k = U_curr - prev_U[b]
                prev_U[b] = U_curr
                is_last = (k == len(step_indices[b]) - 1)
                ppo.rewards[idx] = r_k + (ppo.cfg.L0 if is_last else 0.0) + step_bonus
            else:
                is_last_step = (k == len(step_indices[b]) - 1)
                ppo.rewards[idx] = (ppo.cfg.L0 if is_last_step else 0.0) + Lk - beta * dk + step_bonus


def _run_ibp_batched_round(feats_batch, B, labels, order_inds_batch, omega_batch, ppo, index_to_action,
                           ibp_quantizer, ibp_mod, decoder, current_snr, _ibp_diagnostic_done, decoder_cuda1=None):
    """IBP 批量前向。order_inds_batch[b] 为样本 b 的特征步进顺序（固定 0..D-1），omega_batch [B, D] 为每样本重要性。
    若 decoder_cuda1 非空则每轮同步参数并传入批量部分奖励以双 GPU 并行解码。"""
    ppo.omega_batch_for_update = omega_batch
    ppo.initial_imp_budget_for_update = 333
    ppo.initial_nsk_budget_for_update = 667
    pi_val, pu_val = _compute_pi_pu_for_state(current_snr, ibp_mod.alpha)
    ppo.pi_for_update = pi_val
    ppo.pu_for_update = pu_val
    initial_imp_budget, initial_nsk_budget = 333, 667
    total_budget = 1000
    remaining = [total_budget] * B
    remaining_imp = [initial_imp_budget] * B
    remaining_nsk = [initial_nsk_budget] * B
    bit_allocation = [[] for _ in range(B)]
    imp_stream_list = [[] for _ in range(B)]
    nsk_stream_list = [[] for _ in range(B)]
    total_used_imp_bits = [0] * B
    total_used_nsk_bits = [0] * B
    total_used_bits = [0] * B
    step_imp_bonuses = [[] for _ in range(B)]
    step_indices = [[] for _ in range(B)]
    step_feat_indices = [[] for _ in range(B)]
    step_legal_count = [[] for _ in range(B)]  # 每步后当前合法分配数，用于论文 b(i) 部分分配
    omega_dev = omega_batch.to(feats_batch.device)
    for t in range(512):
        active = [b for b in range(B) if remaining[b] > 0]
        if not active:
            break
        feats_col = torch.stack([feats_batch[b, order_inds_batch[b][t]] for b in active])
        omega_active = omega_dev[active]
        states_tensor = build_state(
            feats_col, omega_active,
            [remaining[b] for b in active], [remaining_imp[b] for b in active], [remaining_nsk[b] for b in active],
            1000, initial_imp_budget, initial_nsk_budget, pi=ppo.pi_for_update, pu=ppo.pu_for_update)
        actions, logps, values = ppo.select_batch(states_tensor, [remaining[b] for b in active],
                                                  [remaining_imp[b] for b in active],
                                                  [remaining_nsk[b] for b in active], index_to_action)
        # 收集需量化的样本，批量量化（避免 B×512 次标量 GPU 调用）
        to_quant_j, to_quant_b, to_quant_i, to_quant_tb, to_quant_ib = [], [], [], [], []
        for j, b in enumerate(active):
            i = order_inds_batch[b][t]
            action_idx = actions[j]
            total_bits, imp_bits = index_to_action[action_idx]
            nsk_bits = total_bits - imp_bits
            if total_bits == 0 or imp_bits > remaining_imp[b] or nsk_bits > remaining_nsk[b]:
                zb = float(getattr(ppo.cfg, "zero_bit_penalty", 0.0))
                # (0,0) 或非法（超 imp/nsk 预算）都视为“本步不发”，在 remaining>0 时惩罚
                zero_pen = -zb * (remaining[b] / float(total_budget)) if (zb > 0 and remaining[b] > 0) else 0.0
                ppo.store_compact(b, feats_batch[b, i].item(), remaining[b], remaining_imp[b], remaining_nsk[b],
                                  action_idx, logps[j], zero_pen, values[j], False, ppo.cfg.action_max_per_step)
                step_indices[b].append(len(ppo.states) - 1)
                step_feat_indices[b].append(i)
                bit_allocation[b].append((i, 0, 0, 0))
                step_legal_count[b].append(len(bit_allocation[b]))
            else:
                to_quant_j.append(j)
                to_quant_b.append(b)
                to_quant_i.append(i)
                to_quant_tb.append(total_bits)
                to_quant_ib.append(imp_bits)
        if to_quant_j:
            feats_to_q = torch.stack([feats_batch[to_quant_b[k], to_quant_i[k]] for k in range(len(to_quant_j))])
            int_vals, _ = ibp_quantizer.quantize_with_ibp_batch(feats_to_q, to_quant_tb, to_quant_ib)
            for k in range(len(to_quant_j)):
                j, b, i = to_quant_j[k], to_quant_b[k], to_quant_i[k]
                total_bits, imp_bits = to_quant_tb[k], to_quant_ib[k]
                nsk_bits = total_bits - imp_bits
                int_val = int_vals[k].item()
                imp_val = int_val >> nsk_bits if imp_bits > 0 else 0
                nsk_val = int_val & ((1 << nsk_bits) - 1) if nsk_bits > 0 else 0
                # 无论 total_bits 多大，都按“重要流/非重要流”顺序写入，比特流会自动跨多个 6-bit 符号打包
                for jj in range(imp_bits):
                    imp_stream_list[b].append(int((imp_val >> (imp_bits - 1 - jj)) & 1))
                for jj in range(nsk_bits):
                    nsk_stream_list[b].append(int((nsk_val >> (nsk_bits - 1 - jj)) & 1))
                step_imp_bonuses[b].append(0.0)
                action_idx = actions[j].item() if isinstance(actions[j], torch.Tensor) else actions[j]
                ppo.store_compact(b, feats_batch[b, i].item(), remaining[b], remaining_imp[b], remaining_nsk[b],
                                  action_idx, logps[j], 0.0, values[j], False, ppo.cfg.action_max_per_step)
                remaining[b] -= total_bits
                remaining_imp[b] -= imp_bits
                remaining_nsk[b] -= nsk_bits
                total_used_imp_bits[b] += imp_bits
                total_used_nsk_bits[b] += nsk_bits
                total_used_bits[b] += total_bits
                step_indices[b].append(len(ppo.states) - 1)
                step_feat_indices[b].append(i)
                bit_allocation[b].append((i, total_bits, imp_bits, nsk_bits))
                step_legal_count[b].append(len(bit_allocation[b]))
    # 终点剩余比特惩罚：鼓励用满预算（加到每图最后一步 reward 上）
    lb = float(getattr(ppo.cfg, "leftover_bit_penalty", 0.0))
    if lb > 0:
        for b in range(B):
            if remaining[b] > 0 and step_indices[b]:
                idx_last = step_indices[b][-1]
                ppo.rewards[idx_last] = float(ppo.rewards[idx_last]) - lb * (remaining[b] / float(total_budget))
    # 批量处理最终重建以提高GPU利用率（所有样本一起解码）
    list_final_rewards, list_task_perf, list_dist, list_imp_ratio = [], [], [], []
    rec_features_batch = torch.zeros_like(feats_batch)  # [B, 512]

    # IBP诊断（只做一次）
    if not _ibp_diagnostic_done[0] and B > 0:
        _ibp_diagnostic_done[0] = True
        b = 0
        alloc_indices = [a[0] for a in bit_allocation[b] if (a[1] if len(a) >= 2 else 0) > 0]
        om_b = omega_batch[b]
        top_n = min(332, om_b.shape[0])
        top_by_omega = torch.argsort(om_b, descending=True)[:top_n].cpu().tolist()
        overlap = len(set(alloc_indices) & set(top_by_omega))
        print(
            f"  [IBP诊断] DPPO 按维序 0..D-1 步进；得到比特的特征数={len(alloc_indices)}, "
            f"alloc 索引前10={alloc_indices[:10]}, ω top-{top_n} 前10={top_by_omega[:10]}, 集合重叠={overlap}/{top_n}（仅供参考）")

    # 批量调制和解调（双流：2 重要 + 4 非重要/符号）
    non_empty_mask = torch.tensor([len(imp_stream_list[b]) > 0 or len(nsk_stream_list[b]) > 0 for b in range(B)],
                                  device=device)
    if non_empty_mask.any():
        bits_batch = pack_ibp_dual_stream_batch(imp_stream_list, nsk_stream_list, device=device)  # [B, pad_to]
        symbols_batch = ibp_mod.modulate(bits_batch, snr_db=current_snr)
        demod_bits_batch = ibp_mod.demodulate(symbols_batch)  # [B, pad_to]

        # 批量重建特征（双流解包）
        for b in range(B):
            if bit_allocation[b]:
                unpacked = unpack_ibp_dual_stream(demod_bits_batch, bit_allocation[b], batch_idx=b, device=device)
                for feat_idx, int_val, total_bits in unpacked:
                    rec_features_batch[b, feat_idx] = dequantize_from_int(int_val, total_bits, device_=device).squeeze()
            # 未分配比特的样本保持 rec 为 0，不填原特征
    else:
        pass  # 无任何样本分配比特时，保持 rec 全 0

    # 批量解码（所有样本一起）；GPU 上用 AMP 加速 decoder 前向
    with torch.no_grad():
        if rec_features_batch.is_cuda:
            with torch.amp.autocast('cuda'):
                logits_batch = decoder(rec_features_batch)  # [B, num_classes]
            logits_batch = logits_batch.float()
        else:
            logits_batch = decoder(rec_features_batch)  # [B, num_classes]
        pred_batch = logits_batch.argmax(dim=1)  # [B]
        task_perf_batch = (pred_batch == labels.squeeze()).float()  # [B]

        # 批量计算distortion（向量化操作）
        diff = feats_batch - rec_features_batch  # [B, 512]
        dist_batch = (omega_batch * (diff ** 2)).sum(dim=1)  # [B]

        # 批量计算imp_usage_ratio（向量化操作）
        total_used_bits_tensor = torch.tensor(total_used_bits, device=device, dtype=torch.float32)
        total_used_imp_bits_tensor = torch.tensor(total_used_imp_bits, device=device, dtype=torch.float32)
        imp_ratio_batch = total_used_imp_bits_tensor / torch.clamp(total_used_bits_tensor, min=1.0)  # [B]

        # 批量计算reward
        final_reward_batch = ppo.cfg.L0 + task_perf_batch - ppo.cfg.beta * dist_batch

        list_final_rewards = final_reward_batch.cpu().tolist()
        list_task_perf = task_perf_batch.cpu().tolist()
        list_dist = dist_batch.cpu().tolist()
        list_imp_ratio = imp_ratio_batch.cpu().tolist()

    # 论文每步奖励 r_i = L0 + L(b(i)) − β·d(b(i))。512 步部分重建+解码一次性批量计算（一次 IBP 调制/解调 + 一次 decoder），提高 GPU 利用率
    max_steps = max(len(step_indices[b]) for b in range(B)) if B else 0
    if max_steps > 0:
        if decoder_cuda1 is not None:
            decoder_cuda1.load_state_dict(decoder.state_dict())
        _ibp_partial_rewards_batched_all(
            feats_batch, B, imp_stream_list, nsk_stream_list, bit_allocation, step_indices, step_legal_count,
            step_imp_bonuses,
            omega_batch, labels, ppo, ibp_quantizer, ibp_mod, decoder, current_snr, decoder_cuda1=decoder_cuda1,
        )
    # 折扣回报用于 PPO；无折扣 Σ r_t 用于绘图（Fig.7 纵轴约 500~3500）
    gamma = getattr(ppo.cfg, "gamma", 0.99)
    list_discounted_returns = []
    list_undiscounted_returns = []
    for b in range(B):
        if not step_indices[b]:
            list_discounted_returns.append(0.0)
            list_undiscounted_returns.append(0.0)
            continue
        G_disc = sum(
            (gamma ** k) * ppo.rewards[step_indices[b][k]]
            for k in range(len(step_indices[b]))
            if step_indices[b][k] < len(ppo.rewards)
        )
        G_undisc = sum(
            ppo.rewards[step_indices[b][k]]
            for k in range(len(step_indices[b]))
            if step_indices[b][k] < len(ppo.rewards)
        )
        list_discounted_returns.append(float(G_disc))
        list_undiscounted_returns.append(float(G_undisc))
    ppo._step_indices_last = step_indices
    ppo._B_last = B
    ppo._last_bit_allocation = bit_allocation  # 供 alpha 优化使用
    return list_final_rewards, list_task_perf, list_dist, list_imp_ratio, list_discounted_returns, list_undiscounted_returns


def _save_ibp_alpha(ibp_mod, alpha_by_snr: dict, snr_list: list, fixed_snr: float):
    """保存 IBP alpha：课程学习时存 alpha_by_snr，否则存 alpha。"""
    if ibp_mod is None:
        return
    data = {"alpha": ibp_mod.alpha}
    if alpha_by_snr:
        data["alpha_by_snr"] = {float(k): float(v) for k, v in alpha_by_snr.items()}
        if fixed_snr in alpha_by_snr:
            data["alpha"] = float(alpha_by_snr[fixed_snr])
    torch.save(data, "ibp_alpha.pth")


def train_dppo_mode(encoder, decoder, omega, train_loader, mode='qam', epochs=20, max_bits=6,
                    batch_size=None, resume_from_10db=0, additional_epochs=0, omega_per_image=None,
                    features_per_image=None,
                    use_per_image_omega_train: bool = True, iterations_per_epoch: int = 32,
                    alpha_ema_tau: float = 0.1, low_memory_mode: bool = True,
                    use_curriculum: Optional[bool] = None, alpha_update_interval: Optional[int] = None):
    mode = str(mode).lower() if mode else 'qam'  # 统一小写，确保走 batched 路径
    """
    训练指定模式的DPPO - 使用真实信道调制解调。
    若提供 omega_per_image [N_train, 512]，则使用每图语义重要性；train_loader 需返回 (imgs, labels, indices)。

    Args:
        encoder: 语义编码器
        decoder: 语义解码器
        omega: 重要性权重 [512]（全局，用于评估或当无 omega_per_image 时）
        train_loader: 训练数据加载器；若 omega_per_image 非空，需为 DatasetWithIndex 的 loader，返回 (imgs, labels, indices)
        omega_per_image: 可选，[N_train, 512] 每图语义重要性，有则训练时按图取用
        features_per_image: 可选，[N_train, 512] 预计算的语义特征，有则训练时按 indices 取用，跳过 encoder。
            传 None 时对 batch 图像在线 encoder；此时应使用带 RandomResizedCrop 等增强的 train_loader，与固定特征相比更不易过拟合。
        mode: 'qam' 或 'ibp'
        epochs: 训练 epoch 数（每 epoch 含 iterations_per_epoch 次迭代）。论文 E_p=20。
        iterations_per_epoch: 每个 epoch 内的迭代次数（每次迭代=1 batch + 1 次 PPO update）。TensorBoard 横轴为 iteration。
        max_bits: 最大比特数
        batch_size: 每轮使用的图片数（默认由 images_per_round 决定）
        use_curriculum: 是否使用SNR课程学习
        snr_schedule: SNR课程学习计划，如果为None则使用默认计划
        resume_from_10db: 0=若 ppo_actor/critic 存在则仅加载并跳过训练；1=加载检查点后继续训练
        additional_epochs: resume_from_10db=1 且 >0 时作为继续训练的 epoch 数，否则继续训练仍用 epochs
        alpha_update_interval: IBP 模式下 alpha 更新间隔（每 N 次迭代同步更新一次，默认 20 以降低环境非平稳性）
        alpha_ema_tau: IBP alpha EMA 更新系数（0~1），越大 alpha 逼近 opt_alpha 越快，默认 0.1
    """
    images_per_round = 2000  # 大批量加速，内存紧张时减小
    _use_curriculum = use_curriculum if use_curriculum is not None else False
    _alpha_update_interval = alpha_update_interval if alpha_update_interval is not None else 5

    # 启用 cuDNN benchmark 以加速卷积等算子（输入尺寸固定时有效）
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 课程学习：-6 到 4 dB，步长 2，均分训练步数
    CURRICULUM_SNR_LIST = [-6.0, -4.0, -2.0, 0.0, 2.0, 4.0]
    FIXED_SNR = -6.0

    if mode == 'ibp':
        snr_list = CURRICULUM_SNR_LIST if _use_curriculum else [FIXED_SNR]
        print(f"\n{'=' * 70}")
        print(f"【IBP模式】开始训练...")
        print(f"{'=' * 70}")
        print(f"  课程学习: {'启用 (-6~4dB, 步长2)' if _use_curriculum else '禁用 (固定4dB)'}")
        print(f"  alpha 更新: 每 {_alpha_update_interval} iter")
    else:
        snr_list = CURRICULUM_SNR_LIST if _use_curriculum else [FIXED_SNR]
        print(f"\n{'=' * 70}")
        print(f"【{mode.upper()}模式】开始训练...")
        print(f"{'=' * 70}")
        print(f"  课程学习: {'启用 (-6~4dB, 步长2)' if _use_curriculum else '禁用 (固定4dB)'}")

    quantizer = DitherQuantizer(gamma=1.0)
    ibp_quantizer = IBPQuantizer(gamma=1.0) if mode == 'ibp' else None
    qam_mod = QAM64()
    IBP_ALPHA_INIT = 0.5
    ibp_mod = IBPMapper(total_bits=6, imp_bits=2, alpha=IBP_ALPHA_INIT,
                        use_fixed_alpha=False) if mode == 'ibp' else None

    # IBP：alpha 可优化；解析步用 NSV 风格权重（omega·批内方差·比特耦合）+ 理论 MSE 目标（见 compute_optimal_alpha_analytical）
    # 动作空间数量（含(0,0)），用于 actor 输出维度
    ibp_action_size_max = (max_bits + 1) * (max_bits + 2) // 2  # 含(0,0)

    # 初始化TensorBoard writer（QAM模式使用，IBP模式在循环内为每个SNR单独初始化）
    if mode != 'ibp':
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/{mode}_{'curriculum' if _use_curriculum else 'snr4'}_{timestamp}"
        writer = SummaryWriter(log_dir)

    if mode == 'ibp':
        ppo_cfg = PPOConfig(
            action_max_per_step=max_bits,
            use_ibp=True,
            state_dim=1 + 512 + 3 + 2,
            use_per_image_omega_train=use_per_image_omega_train,
            action_space_size_ibp=ibp_action_size_max,
            alpha_ema_tau=alpha_ema_tau,
            alpha_update_interval=_alpha_update_interval,
            alpha_use_nsv_weights=True,
            alpha_nsv_bit_coupling=0.2,
            alpha_nsv_weight_scheme="406",
            use_curriculum=_use_curriculum,
            low_memory_mode=low_memory_mode,
            use_dual_gpu_decoder=not low_memory_mode,
        )
        index_to_action, _ = build_ibp_action_space_for_snr(5.0, max_bits)
    else:
        ppo_cfg = PPOConfig(
            action_max_per_step=max_bits,
            use_ibp=False,
            state_dim=1 + 512 + 3,
            use_per_image_omega_train=use_per_image_omega_train,
            use_curriculum=_use_curriculum,
            low_memory_mode=low_memory_mode,
            use_dual_gpu_decoder=not low_memory_mode,
        )
        index_to_action = None

    # 预计算 omega 的 numpy 数组（下标顺序 [omega[0],...,omega[511]]，评估时须与训练一致）
    omega_np = omega.detach().float().cpu().numpy()
    # DPPO 特征步进顺序：固定 0..D-1（与 encoder 维序一致）；omega_batch 仅进入状态向量供策略参考

    # IBP模式：课程学习时在多个SNR上均分步数，否则固定4dB
    if mode == 'ibp':
        all_training_rewards = []
        decoder_cuda1 = None
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2 and getattr(ppo_cfg, 'use_dual_gpu_decoder',
                                                                                    False):
            try:
                decoder_cuda1 = copy.deepcopy(decoder).to('cuda:1')
            except Exception:
                decoder_cuda1 = None

        # 新版 checkpoint：包含 optimizer / 训练进度 / RNG；按 max_bits 区分，避免 6-bit 权重误用于 10-bit
        ckpt_path = _ppo_checkpoint_path(mode, ppo_cfg.action_max_per_step)
        actor_ibp_path = f"ppo_actor_{mode}.pth"   # 兼容旧文件
        critic_ibp_path = f"ppo_critic_{mode}.pth" # 兼容旧文件
        index_to_action, _ = build_ibp_action_space_for_snr(5.0, max_bits)

        ibp_alpha_by_snr = {}

        actor_abs = os.path.abspath(actor_ibp_path)
        critic_abs = os.path.abspath(critic_ibp_path)
        ibp_ckpt_ok = os.path.exists(ckpt_path) or (os.path.exists(actor_abs) and os.path.exists(critic_abs))

        global_iteration = 0
        if ibp_ckpt_ok:
            ppo = PPOAgent(ppo_cfg)
            loaded_ok, loaded_it = _try_load_ppo_checkpoint(ckpt_path, ppo)
            if loaded_ok:
                global_iteration = int(loaded_it)
                print(f"  ✓ 加载 DPPO checkpoint: {os.path.basename(ckpt_path)} (iter={global_iteration})")
            else:
                # 兼容旧版：仅加载 actor/critic（无法恢复 optimizer 动量）
                if os.path.exists(actor_abs) and os.path.exists(critic_abs):
                    ppo.actor.load_state_dict(torch.load(actor_abs, map_location=device, weights_only=True))
                    ppo.critic.load_state_dict(torch.load(critic_abs, map_location=device, weights_only=True))
                    print(f"  ✓ 兼容加载旧权重: {os.path.basename(actor_abs)}（未恢复optimizer，续训可能不如完整ckpt）")
            if os.path.exists(os.path.abspath("ibp_alpha.pth")):
                alpha_data = torch.load(os.path.abspath("ibp_alpha.pth"), weights_only=True)
                loaded_alpha = alpha_data.get("alpha", 1)
                ibp_mod._build_constellation(loaded_alpha)
                alpha_by_snr = alpha_data.get("alpha_by_snr", {})
                if alpha_by_snr:
                    print(f"  ✓ 加载 IBP alpha_by_snr={list(alpha_by_snr.keys())}")
                else:
                    print(f"  ✓ 加载 IBP alpha={loaded_alpha:.4f}")
            rew_ibp = os.path.abspath(f"rewards_{mode}.pth")
            if os.path.exists(rew_ibp):
                _lr = torch.load(rew_ibp, weights_only=False)
                ppo.training_rewards = list(_lr) if isinstance(_lr, (list, tuple)) else list(_lr)
            else:
                ppo.training_rewards = []
            if int(resume_from_10db) == 0:
                print(f"  ✓ 发现 IBP 模型，resume_from_10db=0，跳过训练")
                return ppo, ppo.training_rewards
            training_epochs = int(additional_epochs) if int(additional_epochs) > 0 else int(epochs)
            print(
                f"  ✓ 发现 IBP 模型，resume_from_10db=1：加载后继续训练 {training_epochs} epochs "
                f"(additional_epochs={additional_epochs}, 若为 0 则用 epochs={epochs})"
            )
        else:
            ppo = PPOAgent(ppo_cfg)
            training_epochs = int(epochs)
        total_iterations = training_epochs * iterations_per_epoch
        iters_per_snr = max(1, total_iterations // len(snr_list))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir_snr = f"runs/{mode}_{'curriculum' if _use_curriculum else 'snr4'}_{timestamp}"
        writer_snr = SummaryWriter(log_dir_snr)
        _ibp_diagnostic_done = [False]

        if omega_per_image is not None:
            n_train = len(train_loader.dataset)
            if omega_per_image.shape[0] != n_train:
                raise ValueError(f"omega_per_image 与 train_loader 不匹配")
        train_iter = iter(train_loader)
        pbar = tqdm(range(total_iterations), desc=f"IBP {'Curriculum' if _use_curriculum else '4dB'} Training")

        for training_snr in snr_list:
            current_snr = training_snr
            if _use_curriculum and mode == 'ibp':
                ibp_mod._build_constellation(IBP_ALPHA_INIT)
            for _ in range(iters_per_snr):
                encoder.eval()
                decoder.eval()

                with torch.inference_mode():
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        batch = next(train_iter)

                    if len(batch) == 3:
                        imgs, labels, indices = batch[0], batch[1], batch[2]
                        if torch.is_tensor(indices):
                            indices = indices.cpu().tolist()
                    else:
                        imgs, labels, indices = batch[0], batch[1], None
                    imgs = imgs[:images_per_round].to(device, non_blocking=True)
                    labels = labels[:images_per_round].to(device, non_blocking=True)
                    if features_per_image is not None and indices is not None:
                        idx_list = indices[:images_per_round]
                        if torch.is_tensor(idx_list):
                            idx_list = idx_list.cpu().tolist()
                        feats_batch = features_per_image[idx_list].to(device, non_blocking=True)
                    elif device.type == "cuda":
                        with torch.amp.autocast('cuda'):
                            feats_batch = encoder(imgs)  # [B, 512]
                        feats_batch = feats_batch.float()
                    else:
                        feats_batch = encoder(imgs)  # [B, 512]
                    del imgs  # 释放图像显存，仅保留特征
                B = feats_batch.size(0)
                use_per_img = getattr(ppo_cfg, "use_per_image_omega_train", True)
                _visit = _dppo_feature_visit_order(feats_batch.shape[1])
                order_inds_batch = [_visit] * B
                if omega_per_image is not None and indices is not None and use_per_img:
                    idx_t = torch.tensor(indices[:B], device=omega_per_image.device)
                    omega_batch = omega_per_image[idx_t].to(device)
                else:
                    omega_batch = omega.unsqueeze(0).expand(B, -1)

                list_final_rewards = []
                list_task_perf = []
                list_dist = []
                list_imp_ratio = []
                list_discounted_returns = []
                list_undiscounted_returns = []

                list_final_rewards, list_task_perf, list_dist, list_imp_ratio, list_discounted_returns, list_undiscounted_returns = _run_ibp_batched_round(
                    feats_batch, B, labels, order_inds_batch, omega_batch, ppo, index_to_action,
                    ibp_quantizer, ibp_mod, decoder, current_snr, _ibp_diagnostic_done, decoder_cuda1=decoder_cuda1)

                do_alpha = False
                if mode == 'ibp' and ibp_mod is not None and hasattr(ppo, '_last_bit_allocation'):
                    bit_alloc = getattr(ppo, '_last_bit_allocation', None)
                    alpha_int = getattr(ppo.cfg, "alpha_update_interval", 1)
                    do_alpha = (omega_batch is not None and bit_alloc is not None and
                                global_iteration % alpha_int == 0 and _SER_AVAILABLE)
                    if do_alpha:
                        omega_copy = omega_batch.detach().cpu().clone()
                        feats_copy = feats_batch.detach().cpu().clone()
                        bit_alloc_copy = copy.deepcopy(bit_alloc)
                        # =============== 【修改点：接入高精度解析求解器】 ===============
                        try:
                            _use_nsv = getattr(ppo.cfg, "alpha_use_nsv_weights", True)
                            _gamma = float(getattr(ppo.cfg, "alpha_nsv_bit_coupling", 0.2))
                            _scheme = getattr(ppo.cfg, "alpha_nsv_weight_scheme", "406")
                            opt_alpha = compute_optimal_alpha_analytical(
                                omega_batch=omega_copy,
                                bit_allocation_batch=bit_alloc_copy,
                                current_snr=current_snr,
                                feats_batch=feats_copy if _use_nsv else None,
                                decoder=decoder if _use_nsv else None,
                                use_nsv_weights=_use_nsv,
                                nsv_bit_coupling=_gamma,
                                nsv_weight_scheme=_scheme,
                            )
                        except Exception as e:
                            # 容错兜底：如果极小概率出现计算异常，目标 alpha 保持不变
                            opt_alpha = ibp_mod.alpha
                        # ================================================================

                ppo.update()
                if do_alpha:
                    old_alpha = ibp_mod.alpha
                    opt_alpha_clipped = max(0.0, min(1.16, opt_alpha))
                    tau = getattr(ppo.cfg, "alpha_ema_tau", 0.1)
                    new_alpha = tau * opt_alpha_clipped + (1.0 - tau) * old_alpha
                    new_alpha = max(0.0, min(1.16, new_alpha))
                    if opt_alpha >= 1.15:
                        _debug_alpha_bit_distribution(bit_alloc)
                    if abs(new_alpha - old_alpha) > 1e-6:
                        ibp_mod._build_constellation(new_alpha)
                        print(f"  [Alpha] {old_alpha:.4f} -> {new_alpha:.4f} (opt={opt_alpha:.4f}, tau={tau})")
                if list_final_rewards:
                    avg_undisc = np.mean(list_undiscounted_returns) if list_undiscounted_returns else np.mean(
                        list_final_rewards)
                    ppo.training_rewards.append(avg_undisc)
                    writer_snr.add_scalar('Reward', avg_undisc, global_iteration)
                    writer_snr.add_scalar('Task_Perf', np.mean(list_task_perf), global_iteration)
                    writer_snr.add_scalar('Distortion', np.mean(list_dist), global_iteration)
                    writer_snr.add_scalar('IBP/Alpha', ibp_mod.alpha, global_iteration)
                    if list_imp_ratio:
                        writer_snr.add_scalar('IBP/Imp_Ratio', np.mean(list_imp_ratio), global_iteration)

                global_iteration += 1
                pbar.update(1)
                if list_final_rewards:
                    _r = np.mean(list_undiscounted_returns) if list_undiscounted_returns else np.mean(
                        list_final_rewards)
                    pbar.set_postfix({'SNR': f'{current_snr}dB', 'iter': global_iteration, 'Reward': f'{_r:.1f}'})

                if global_iteration % 100 == 0 or global_iteration == 1:
                    _save_ppo_checkpoint(
                        ckpt_path, ppo,
                        global_iteration=global_iteration,
                        training_rewards=ppo.training_rewards,
                        extra={"snr": float(current_snr), "mode": "ibp"},
                    )
                    # 同时保存旧权重文件，便于外部脚本兼容
                    torch.save(ppo.actor.state_dict(), actor_ibp_path)
                    torch.save(ppo.critic.state_dict(), critic_ibp_path)
                    _save_ibp_alpha(ibp_mod, ibp_alpha_by_snr, snr_list, FIXED_SNR)

                del feats_batch, omega_batch, order_inds_batch, list_final_rewards, list_task_perf, list_dist
                del list_discounted_returns, list_undiscounted_returns, list_imp_ratio
                if low_memory_mode and global_iteration % 20 == 0 and device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            # 该 SNR 训练结束，记录 alpha
            if mode == 'ibp' and ibp_mod is not None:
                ibp_alpha_by_snr[training_snr] = float(ibp_mod.alpha)

        # IBP 训练全部结束，保存
        _save_ppo_checkpoint(
            ckpt_path, ppo,
            global_iteration=global_iteration,
            training_rewards=ppo.training_rewards,
            extra={"snr": float(FIXED_SNR), "mode": "ibp"},
        )
        torch.save(ppo.actor.state_dict(), actor_ibp_path)
        torch.save(ppo.critic.state_dict(), critic_ibp_path)
        _save_ibp_alpha(ibp_mod, ibp_alpha_by_snr, snr_list, FIXED_SNR)
        torch.save(ppo.training_rewards, f"rewards_{mode}.pth")
        writer_snr.close()
        print(f"\n✓ IBP模型训练完成")
        return ppo, ppo.training_rewards

    else:
        # QAM模式：保持原有逻辑（固定SNR=4dB）
        ppo = PPOAgent(ppo_cfg)
        index_to_action = None
        # 双 GPU decoder：一次 deepcopy 到 cuda:1，每轮 load_state_dict 同步，用于批量部分奖励
        decoder_cuda1 = None
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2 and getattr(ppo_cfg, 'use_dual_gpu_decoder',
                                                                                    False):
            try:
                decoder_cuda1 = copy.deepcopy(decoder).to('cuda:1')
            except Exception:
                decoder_cuda1 = None

        # 定义模型路径
        ckpt_path = _ppo_checkpoint_path(mode, ppo_cfg.action_max_per_step)
        actor_path = f"ppo_actor_{mode}.pth"   # 兼容旧文件
        critic_path = f"ppo_critic_{mode}.pth" # 兼容旧文件

        actor_abs = os.path.abspath(actor_path)
        critic_abs = os.path.abspath(critic_path)
        qam_ckpt_ok = os.path.exists(ckpt_path) or (os.path.exists(actor_abs) and os.path.exists(critic_abs))
        global_iteration = 0
        if qam_ckpt_ok:
            print(f"✓ 发现已保存的{mode.upper()}模型，加载中...")
            loaded_ok, loaded_it = _try_load_ppo_checkpoint(ckpt_path, ppo)
            if loaded_ok:
                global_iteration = int(loaded_it)
                print(f"  ✓ 加载 DPPO checkpoint: {os.path.basename(ckpt_path)} (iter={global_iteration})")
            else:
                if os.path.exists(actor_abs) and os.path.exists(critic_abs):
                    ppo.actor.load_state_dict(torch.load(actor_abs, map_location=device, weights_only=True))
                    ppo.critic.load_state_dict(torch.load(critic_abs, map_location=device, weights_only=True))
                    print(f"  ✓ 兼容加载旧权重: {os.path.basename(actor_abs)}（未恢复optimizer，续训可能不如完整ckpt）")
            rew_path = os.path.abspath(f"rewards_{mode}.pth")
            if os.path.exists(rew_path):
                _lr = torch.load(rew_path, weights_only=False)
                ppo.training_rewards = list(_lr) if isinstance(_lr, (list, tuple)) else list(_lr)
            else:
                ppo.training_rewards = []
            if int(resume_from_10db) == 0:
                writer.close()
                return ppo, ppo.training_rewards
            training_epochs = int(additional_epochs) if int(additional_epochs) > 0 else int(epochs)
            print(
                f"  resume_from_10db=1：在已有权重上继续训练 {training_epochs} epochs "
                f"(additional_epochs={additional_epochs}, 若为 0 则用 epochs={epochs})"
            )
        else:
            ppo.training_rewards = []
            training_epochs = int(epochs)
            print(f"✗ 未找到{mode.upper()}模型，开始训练...")

        _ibp_diagnostic_done = None

        total_iterations = training_epochs * iterations_per_epoch
        iters_per_snr = max(1, total_iterations // len(snr_list))
        print(f"  总迭代次数: {total_iterations}，课程SNR: {snr_list}，每SNR迭代: {iters_per_snr}")
        pbar = tqdm(range(total_iterations),
                    desc=f"{mode.upper()} {'Curriculum' if _use_curriculum else '4dB'} Training")

        if omega_per_image is not None:
            n_train = len(train_loader.dataset)
            if omega_per_image.shape[0] != n_train:
                raise ValueError(
                    f"train_dppo_mode: omega_per_image 与 train_loader 不匹配，"
                    f"omega 行数={omega_per_image.shape[0]}，训练集样本数={n_train}。"
                    "请勿将测试集 omega 用于训练或反之。"
                )
        train_iter = iter(train_loader)

        for training_snr in snr_list:
            current_snr = training_snr
            for _ in range(iters_per_snr):
                encoder.eval()
                decoder.eval()

                # 每迭代取一个 batch；若使用每图 omega 则 batch 为 (imgs, labels, indices)
                with torch.inference_mode():
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(train_loader)
                        batch = next(train_iter)

                    if len(batch) == 3:
                        imgs, labels, indices = batch[0], batch[1], batch[2]
                        if torch.is_tensor(indices):
                            indices = indices.cpu().tolist()
                    else:
                        imgs, labels, indices = batch[0], batch[1], None
                    imgs = imgs[:images_per_round].to(device, non_blocking=True)
                    labels = labels[:images_per_round].to(device, non_blocking=True)
                    if features_per_image is not None and indices is not None:
                        idx_list = indices[:images_per_round]
                        if torch.is_tensor(idx_list):
                            idx_list = idx_list.cpu().tolist()
                        feats_batch = features_per_image[idx_list].to(device, non_blocking=True)
                    elif device.type == "cuda":
                        with torch.amp.autocast('cuda'):
                            feats_batch = encoder(imgs)  # [B, 512]
                        feats_batch = feats_batch.float()
                    else:
                        feats_batch = encoder(imgs)  # [B, 512]
                    del imgs  # 释放图像显存
                    B = feats_batch.size(0)
                    use_per_img = getattr(ppo_cfg, "use_per_image_omega_train", True)
                    _visit = _dppo_feature_visit_order(feats_batch.shape[1])
                    order_inds_batch = [_visit] * B
                    if omega_per_image is not None and indices is not None and use_per_img:
                        idx_t = torch.tensor(indices[:B], device=omega_per_image.device)
                        omega_batch = omega_per_image[idx_t].to(device)
                    else:
                        omega_batch = omega.unsqueeze(0).expand(B, -1)

                list_final_rewards = []
                list_task_perf = []
                list_dist = []
                list_imp_ratio = [] if mode == 'ibp' else None
                list_discounted_returns = []
                list_undiscounted_returns = []

                if mode == 'ibp':
                    list_final_rewards, list_task_perf, list_dist, list_imp_ratio, list_discounted_returns, list_undiscounted_returns = _run_ibp_batched_round(
                        feats_batch, B, labels, order_inds_batch, omega_batch, ppo, index_to_action,
                        ibp_quantizer, ibp_mod, decoder, current_snr, _ibp_diagnostic_done, decoder_cuda1=decoder_cuda1)
                else:
                    # QAM 或任意非 IBP 模式：统一使用 batched 路径（向量化，避免 512×B 次单步循环）
                    list_final_rewards, list_task_perf, list_dist, list_discounted_returns, list_undiscounted_returns = _run_qam_batched_round(
                        feats_batch, B, labels, order_inds_batch, omega_batch, ppo,
                        quantizer, qam_mod, decoder, current_snr, decoder_cuda1=decoder_cuda1)
                    list_imp_ratio = None
                    ppo.update()

                if list_final_rewards:
                    avg_undisc = np.mean(list_undiscounted_returns) if list_undiscounted_returns else np.mean(
                        list_final_rewards)
                    ppo.training_rewards.append(avg_undisc)
                    writer.add_scalar('Reward', avg_undisc, global_iteration)
                    writer.add_scalar('Task_Perf', np.mean(list_task_perf), global_iteration)
                    writer.add_scalar('Distortion', np.mean(list_dist), global_iteration)
                    if mode == 'ibp' and list_imp_ratio:
                        writer.add_scalar('IBP/Imp_Ratio', np.mean(list_imp_ratio), global_iteration)

                global_iteration += 1
                pbar.update(1)
                if list_final_rewards:
                    _r = np.mean(list_undiscounted_returns) if list_undiscounted_returns else np.mean(
                        list_final_rewards)
                    pbar.set_postfix({'SNR': f'{current_snr}dB', 'iter': global_iteration, 'Reward': f'{_r:.1f}'})

                # 每 100 迭代保存一次，减少 I/O 提速
                if global_iteration % 100 == 0 or global_iteration == 1:
                    _save_ppo_checkpoint(
                        ckpt_path, ppo,
                        global_iteration=global_iteration,
                        training_rewards=ppo.training_rewards,
                        extra={"snr": float(current_snr), "mode": str(mode).lower()},
                    )
                    torch.save(ppo.actor.state_dict(), actor_path)
                    torch.save(ppo.critic.state_dict(), critic_path)

                # 清理迭代中的中间变量（在使用完后）
                del feats_batch, omega_batch, order_inds_batch, list_final_rewards, list_task_perf, list_dist
                del list_discounted_returns, list_undiscounted_returns
                if mode == 'ibp' and list_imp_ratio is not None:
                    del list_imp_ratio
                if low_memory_mode and global_iteration % 20 == 0 and device.type == "cuda":
                    torch.cuda.empty_cache()
                    gc.collect()

    # 保存最终模型和奖励
    _save_ppo_checkpoint(
        ckpt_path, ppo,
        global_iteration=global_iteration,
        training_rewards=ppo.training_rewards,
        extra={"mode": str(mode).lower()},
    )
    torch.save(ppo.actor.state_dict(), actor_path)
    torch.save(ppo.critic.state_dict(), critic_path)
    torch.save(ppo.training_rewards, f"rewards_{mode}.pth")
    print(f"\n✓ {mode.upper()}模型训练完成，已保存")
    print(f"✓ TensorBoard日志保存到: {log_dir}")
    print(f"  查看TensorBoard: tensorboard --logdir=runs")

    # 关闭writer
    writer.close()

    return ppo, ppo.training_rewards


def _qam_eval_batched_alloc(batch_feats: torch.Tensor,
                            order_inds_batch: List[List[int]],
                            omega_batch: torch.Tensor,
                            ppo_snr: PPOAgent,
                            quantizer: "DitherQuantizer",
                            max_bits: int) -> Tuple[
    List[List[Tuple[int, int]]], List[List[int]], List[Dict[str, int]], Dict[int, int]]:
    """QAM 评估阶段批量比特分配：512 步每步 select_batch 一次，向量化策略执行。返回 bit_alloc_batch, bits_batch, per_image_bits_batch, qam_bit_stats_batch。"""
    B = batch_feats.size(0)
    remaining = [1000] * B
    bit_allocation = [[] for _ in range(B)]
    all_bits_list = [[] for _ in range(B)]
    image_total_bits = [0] * B
    qam_bit_stats_batch: Dict[int, int] = {}
    omega_dev = omega_batch.to(batch_feats.device)
    for t in range(512):
        active = [b for b in range(B) if remaining[b] > 0]
        if not active:
            break
        feats_col = torch.stack([batch_feats[b, order_inds_batch[b][t]] for b in active])
        omega_active = omega_dev[active]
        states_tensor = build_state(
            feats_col, omega_active,
            [remaining[b] for b in active], None, None, 1000, None, None)
        actions, _, _ = ppo_snr.select_batch(
            states_tensor, [remaining[b] for b in active], None, None, None)
        to_quant_j, to_quant_b, to_quant_i, to_quant_bits = [], [], [], []
        for j, b in enumerate(active):
            i = order_inds_batch[b][t]
            # 评估动作上限：以策略自身 cfg 为准，避免外部 max_bits 参数仍为 6 导致“看起来被截断”
            cap = int(getattr(ppo_snr.cfg, "action_max_per_step", max_bits))
            action_bits = min(actions[j].item() if isinstance(actions[j], torch.Tensor) else actions[j], remaining[b],
                              cap)
            if action_bits < 0:
                action_bits = 0
            bit_allocation[b].append((i, action_bits))
            if action_bits > 0:
                to_quant_j.append(j)
                to_quant_b.append(b)
                to_quant_i.append(i)
                to_quant_bits.append(action_bits)
                remaining[b] -= action_bits
                image_total_bits[b] += action_bits
            qam_bit_stats_batch[action_bits] = qam_bit_stats_batch.get(action_bits, 0) + 1
        if to_quant_j:
            feats_to_q = torch.stack([batch_feats[to_quant_b[k], to_quant_i[k]] for k in range(len(to_quant_j))])
            int_vals = quantizer.quantize_batch_multi_bits(feats_to_q, to_quant_bits)
            for k in range(len(to_quant_j)):
                b = to_quant_b[k]
                action_bits = to_quant_bits[k]
                int_val = int_vals[k].item()
                for bit_idx in range(action_bits):
                    all_bits_list[b].append(int((int_val >> bit_idx) & 1))
    per_image_bits_batch = [{'total': image_total_bits[b]} for b in range(B)]
    return bit_allocation, all_bits_list, per_image_bits_batch, qam_bit_stats_batch


def _ibp_eval_batched_alloc(batch_feats: torch.Tensor,
                            order_inds_batch: List[List[int]],
                            omega_batch: torch.Tensor,
                            ppo_snr: PPOAgent,
                            index_to_action: Dict[int, Tuple[int, int]],
                            ibp_quantizer: IBPQuantizer,
                            _ibp_diagnostic_done: List[bool],
                            snr_db: float = 5.0, alpha: float = 1.0,
                            include_pi_pu: bool = True) -> Tuple[List[List[Tuple[int, int, int, int]]],
List[List[int]],
List[Dict[str, int]],
Dict[Tuple[int, int], int]]:
    """
    IBP 评估阶段的批量比特分配；order_inds_batch[b] 为固定维序 0..D-1，omega_batch [B, D] 为每样本重要性。
    """
    B = batch_feats.size(0)
    dev = batch_feats.device
    initial_imp_budget, initial_nsk_budget = 333, 667
    remaining = [1000] * B
    remaining_imp = [initial_imp_budget] * B
    remaining_nsk = [initial_nsk_budget] * B
    bit_allocation: List[List[Tuple[int, int, int, int]]] = [[] for _ in range(B)]
    imp_stream_list: List[List[int]] = [[] for _ in range(B)]
    nsk_stream_list: List[List[int]] = [[] for _ in range(B)]
    image_total_bits = [0] * B
    image_imp_bits = [0] * B
    image_nsk_bits = [0] * B
    ibp_bit_stats_batch: Dict[Tuple[int, int], int] = {}

    omega_dev = omega_batch.to(dev)
    for t in range(512):
        active = [b for b in range(B) if remaining[b] > 0]
        if not active:
            break
        feats_col = torch.stack([batch_feats[b, order_inds_batch[b][t]] for b in active])
        omega_active = omega_dev[active]
        pi_val, pu_val = _compute_pi_pu_for_state(snr_db, alpha)
        states_tensor = build_state(
            feats_col, omega_active,
            [remaining[b] for b in active],
            [remaining_imp[b] for b in active],
            [remaining_nsk[b] for b in active],
            1000, initial_imp_budget, initial_nsk_budget, pi=pi_val, pu=pu_val)
        actions, _, _ = ppo_snr.select_batch(
            states_tensor,
            [remaining[b] for b in active],
            [remaining_imp[b] for b in active],
            [remaining_nsk[b] for b in active],
            index_to_action)
        to_quant_j, to_quant_b, to_quant_i, to_quant_tb, to_quant_ib = [], [], [], [], []
        for j, b in enumerate(active):
            i = order_inds_batch[b][t]
            action_idx = actions[j]
            total_bits, imp_bits = index_to_action[action_idx]
            nsk_bits = total_bits - imp_bits
            if total_bits == 0 or imp_bits > remaining_imp[b] or nsk_bits > remaining_nsk[b]:
                continue
            to_quant_j.append(j)
            to_quant_b.append(b)
            to_quant_i.append(i)
            to_quant_tb.append(total_bits)
            to_quant_ib.append(imp_bits)
        if to_quant_j:
            feats_to_q = torch.stack([batch_feats[to_quant_b[k], to_quant_i[k]] for k in range(len(to_quant_j))])
            int_vals, _ = ibp_quantizer.quantize_with_ibp_batch(feats_to_q, to_quant_tb, to_quant_ib)
            for k in range(len(to_quant_j)):
                b, i = to_quant_b[k], to_quant_i[k]
                total_bits, imp_bits = to_quant_tb[k], to_quant_ib[k]
                nsk_bits = total_bits - imp_bits
                int_val = int_vals[k].item()
                imp_val = int_val >> nsk_bits if imp_bits > 0 else 0
                nsk_val = int_val & ((1 << nsk_bits) - 1) if nsk_bits > 0 else 0
                # 与训练一致：无论 total_bits 多大，都将 imp_bits 与 nsk_bits 全部写入双流，
                # 由 pack_ibp_dual_stream_batch 自动跨多个 6-bit 符号打包，避免 total_bits>6 时丢位/截断。
                for jj in range(imp_bits):
                    imp_stream_list[b].append(int((imp_val >> (imp_bits - 1 - jj)) & 1))
                for jj in range(nsk_bits):
                    nsk_stream_list[b].append(int((nsk_val >> (nsk_bits - 1 - jj)) & 1))
                remaining[b] -= total_bits
                remaining_imp[b] -= imp_bits
                remaining_nsk[b] -= nsk_bits
                image_total_bits[b] += total_bits
                image_imp_bits[b] += imp_bits
                image_nsk_bits[b] += nsk_bits
                bit_allocation[b].append((i, total_bits, imp_bits, nsk_bits))
                key = (imp_bits, nsk_bits)
                ibp_bit_stats_batch[key] = ibp_bit_stats_batch.get(key, 0) + 1

    if not _ibp_diagnostic_done[0]:
        for b in range(B):
            if imp_stream_list[b] or nsk_stream_list[b]:
                _ibp_diagnostic_done[0] = True
                alloc_indices = [a[0] for a in bit_allocation[b] if (a[1] if len(a) >= 2 else 0) > 0]
                om_b = omega_batch[b]
                top_n = min(332, om_b.shape[0])
                top_by_omega = torch.argsort(om_b, descending=True)[:top_n].cpu().tolist()
                overlap = len(set(alloc_indices) & set(top_by_omega))
                print(f"  [IBP诊断] DPPO 按维序 0..D-1 步进；得到比特的特征数={len(alloc_indices)}, "
                      f"alloc 索引前10={alloc_indices[:10]}, ω top-{top_n} 前10={top_by_omega[:10]}, 集合重叠={overlap}/{top_n}（仅供参考；重建用 total_bits）")
                break

    per_image_bits = []
    for b in range(B):
        per_image_bits.append({
            'total': image_total_bits[b],
            'imp': image_imp_bits[b],
            'nsk': image_nsk_bits[b],
        })

    # 双流打包：批量向量化
    packed_batch = pack_ibp_dual_stream_batch(imp_stream_list, nsk_stream_list, device=dev)
    bits_batch = [packed_batch[b].cpu().tolist() for b in range(B)]
    return bit_allocation, bits_batch, per_image_bits, ibp_bit_stats_batch


def evaluate_mode(encoder, decoder, omega_per_image_test, test_loader, ppo, mode='qam',
                  max_bits=6, num=1000, snr_values=None, use_snr_models: bool = False,
                  eval_use_training_policy: bool = True,
                  use_per_image_omega_eval: bool = True,
                  features_per_image_test=None):
    """评估指定模式的性能。test_loader 须返回 (imgs, labels, indices)，每张图使用自己的 omega_per_image_test[indices]。
    语义重要性须与训练一致：omega_per_image_test[i] 必须对应 test_set 中第 i 张图（与 compute_omega_per_image 顺序一致），
    且 test_loader 的 indices 须为数据集下标，否则会出现「重要性错位」导致评估准确率明显低于训练时的 top-1 准确率。
    eval_use_training_policy: 若 True，评估时与训练一致使用随机策略 sample()（固定种子可复现），避免 argmax 与训练分布不一致导致准确率骤降。"""
    n_test = len(test_loader.dataset)
    if omega_per_image_test.shape[0] != n_test:
        raise ValueError(
            f"evaluate_mode: omega_per_image_test 与 test_loader 不匹配，"
            f"omega 行数={omega_per_image_test.shape[0]}，测试集样本数={n_test}。"
            "请勿将训练集 omega 用于测试集或反之。"
        )
    if snr_values is None:
        snr_values = list(range(-6, 11, 2))
    quantizer = DitherQuantizer(gamma=1.0)
    ibp_quantizer = IBPQuantizer(gamma=1.0) if mode == 'ibp' else None
    qam_mod = QAM64()
    ibp_alpha_init = 0.5
    ibp_alpha_by_snr = {}
    if mode == 'ibp' and os.path.exists("ibp_alpha.pth"):
        try:
            alpha_data = torch.load("ibp_alpha.pth", map_location="cpu", weights_only=True)
            # ibp_alpha_init = 0.5
            ibp_alpha_by_snr = alpha_data.get("alpha_by_snr", {})
            if ibp_alpha_by_snr:
                print(f"  [Eval] 加载 IBP alpha_by_snr={list(ibp_alpha_by_snr.keys())}")
            else:
                print(f"  [Eval] 加载 IBP alpha={ibp_alpha_init:.4f}")
        except Exception:
            pass
    ibp_mod = IBPMapper(total_bits=6, imp_bits=2, alpha=ibp_alpha_init, use_fixed_alpha=True) if mode == 'ibp' else None

    ibp_action_size_max = (max_bits + 1) * (max_bits + 2) // 2  # 28，含(0,0)，与训练一致

    # 评估时动作上限优先取传入 ppo 的 cfg（若有），避免使用默认 max_bits=6
    if ppo is not None and hasattr(ppo, "cfg") and hasattr(ppo.cfg, "action_max_per_step"):
        try:
            max_bits = int(getattr(ppo.cfg, "action_max_per_step"))
        except Exception:
            pass

    if mode == 'ibp':
        ppo_cfg = PPOConfig(
            action_max_per_step=max_bits,
            use_ibp=True,
            state_dim=1 + 512 + 3 + 2,  # +2: Pi/Pu 误比特率
            use_per_image_omega_eval=use_per_image_omega_eval,
            action_space_size_ibp=ibp_action_size_max,
            alpha_use_nsv_weights=True,
            alpha_nsv_bit_coupling=0.2,
            alpha_nsv_weight_scheme="406",
        )
        index_to_action, _ = build_ibp_action_space_for_snr(5.0, max_bits)
    else:
        # QAM模式：状态维度与IBP统一（1+512），与论文一致
        ppo_cfg = PPOConfig(
            action_max_per_step=max_bits,
            use_ibp=False,
            state_dim=1 + 512 + 3,
            use_per_image_omega_eval=use_per_image_omega_eval,
        )
        index_to_action = None

    accuracies = []
    qam_bit_stats = {}
    ibp_bit_stats = {}
    total_images_evaluated = 0
    all_per_image_bits = []
    _ibp_diagnostic_done = [False]

    # 若使用全局 omega_eval，则在此处对每图 omega 求平均，得到 [512] 全局重要性
    use_per_img_eval = getattr(ppo_cfg, "use_per_image_omega_eval", True)
    if not use_per_img_eval:
        omega_eval_global = omega_per_image_test.mean(dim=0).to(device)
    else:
        omega_eval_global = None
    if eval_use_training_policy and mode == 'ibp':
        print(f"  [Eval] 与训练一致使用随机策略 sample()（固定种子），避免 argmax 导致准确率偏低。")

    # IBP模式：与训练一致，单模型(ppo_actor_ibp.pth)、欧氏动作空间、alpha=0.4，全SNR评估
    # QAM模式：使用传入的ppo或默认模型
    actor_default_path = f"ppo_actor_{mode}.pth"
    critic_default_path = f"ppo_critic_{mode}.pth"
    ckpt_path = _ppo_checkpoint_path(mode, max_bits)
    ppo_snr = ppo
    # 优先加载完整 ckpt（含 cfg/max_bits 对应的权重），其次兼容旧 actor/critic
    if os.path.exists(ckpt_path):
        ppo_snr = PPOAgent(ppo_cfg)
        _ok, _ = _try_load_ppo_checkpoint(ckpt_path, ppo_snr)
        if _ok:
            print(f"  [OK] 加载 DPPO checkpoint: {os.path.basename(ckpt_path)}")
    elif os.path.exists(actor_default_path) and os.path.exists(critic_default_path):
        ppo_snr = PPOAgent(ppo_cfg)
        ppo_snr.actor.load_state_dict(torch.load(actor_default_path, map_location=device))
        ppo_snr.critic.load_state_dict(torch.load(critic_default_path, map_location=device))
        print(f"  [OK] 加载旧模型: {actor_default_path}")
    else:
        print(f"  [INFO] 使用传入的ppo模型进行评估")

    encoder.eval()
    decoder.eval()

    # 多 SNR 缓存：IBP 单模型/单动作空间；QAM 单模型
    ppo_snr_by_snr = {}
    index_to_action_by_snr = {}
    for snr_db in snr_values:
        if mode == 'ibp':
            index_to_action_by_snr[snr_db] = index_to_action  # 欧氏排序，与训练一致
            ppo_snr_by_snr[snr_db] = ppo_snr
        else:
            index_to_action_by_snr[snr_db] = None
            ppo_snr_by_snr[snr_db] = ppo_snr

    correct_per_snr = {s: 0 for s in snr_values}
    total_per_snr = {s: 0 for s in snr_values}
    per_image_bits_by_snr = {s: [] for s in snr_values}
    for _snr in snr_values:
        ppo_snr_by_snr[_snr].critic.eval()
        if eval_use_training_policy:
            ppo_snr_by_snr[_snr].actor.train()
        else:
            ppo_snr_by_snr[_snr].actor.eval()
    if eval_use_training_policy:
        torch.manual_seed(42)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(42)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"{mode.upper()} Eval", total=min(len(test_loader), (num // 32) + 5)):
            if all(total_per_snr[s] >= num for s in snr_values):
                break
            if len(batch) != 3:
                raise ValueError("evaluate_mode 需要 test_loader 返回 (imgs, labels, indices)，请使用 DatasetWithIndex。")
            images, labels, indices = batch[0], batch[1], batch[2]
            if torch.is_tensor(indices):
                indices = indices.cpu().tolist()
            images, labels = images.to(device), labels.to(device)
            if features_per_image_test is not None:
                idx_list = indices[:images.size(0)]
                if torch.is_tensor(idx_list):
                    idx_list = idx_list.cpu().tolist()
                batch_feats = features_per_image_test[idx_list].to(device)
            else:
                batch_feats = encoder(images)  # 缓存：同一 batch 在所有 SNR 下共用，只算一次
            B = batch_feats.size(0)
            if use_per_img_eval:
                idx_t = torch.tensor(indices[:B], device=omega_per_image_test.device)
                omega_batch = omega_per_image_test[idx_t].to(device)
            else:
                omega_batch = omega_eval_global.unsqueeze(0).expand(B, -1)
            _visit = _dppo_feature_visit_order(batch_feats.shape[1])
            order_inds_batch = [_visit] * B

            for snr_db in snr_values:
                if total_per_snr[snr_db] >= num:
                    continue
                ppo_snr = ppo_snr_by_snr[snr_db]
                index_to_action = index_to_action_by_snr[snr_db]
                if mode == 'ibp' and total_per_snr[snr_db] == 0 and B > 0:
                    _om0 = omega_batch[0].cpu().numpy()
                    _om1 = omega_batch[min(1, B - 1)].cpu().numpy()
                    diff_01 = np.abs(_om0 - _om1).sum()
                    print(
                        f"  [Eval诊断] 首 batch indices={indices[:min(3, B)]}..., omega_batch[0]与[1]差和={diff_01:.4f} "
                        f"(应>0 表每图不同), 特征步进顺序=固定 0..{batch_feats.shape[1] - 1}")
                if mode == 'ibp':

                    bit_alloc_batch, bits_batch, per_image_bits_batch, ibp_stats_batch = _ibp_eval_batched_alloc(
                        batch_feats, order_inds_batch, omega_batch, ppo_snr, index_to_action, ibp_quantizer,
                        _ibp_diagnostic_done, snr_db=snr_db, alpha=ibp_mod.alpha)
                    try:
                        _use_nsv = getattr(ppo_snr.cfg, "alpha_use_nsv_weights", True)
                        _gamma = float(getattr(ppo_snr.cfg, "alpha_nsv_bit_coupling", 0.2))
                        _scheme = getattr(ppo_snr.cfg, "alpha_nsv_weight_scheme", "406")
                        # 基于 bit_alloc、SNR 与 406/320 NSV（W1+Sigma+注水）权重求理论 MSE 意义下的最优 alpha
                        opt_alpha = compute_optimal_alpha_analytical(
                            omega_batch=omega_batch.detach().cpu(),
                            bit_allocation_batch=bit_alloc_batch,
                            current_snr=snr_db,
                            feats_batch=batch_feats.detach().cpu() if _use_nsv else None,
                            decoder=decoder if _use_nsv else None,
                            use_nsv_weights=_use_nsv,
                            nsv_bit_coupling=_gamma,
                            nsv_weight_scheme=_scheme,
                        )
                        # 实时更新星座图！接下来的 modulate(调制) 将使用这个最优的新星座图
                        ibp_mod._build_constellation(opt_alpha)
                    except Exception as e:
                        # 容错：如果极小概率求解失败，维持之前的 alpha 不变
                        pass
                    for key, v in ibp_stats_batch.items():
                        ibp_bit_stats[key] = ibp_bit_stats.get(key, 0) + v
                else:
                    bit_alloc_batch, bits_batch, per_image_bits_batch, qam_stats_batch = _qam_eval_batched_alloc(
                        batch_feats, order_inds_batch, omega_batch, ppo_snr, quantizer, max_bits)
                    for k, v in qam_stats_batch.items():
                        qam_bit_stats[k] = qam_bit_stats.get(k, 0) + v

                if mode == 'ibp':
                    bits_tensors = []
                    max_len = 0
                    for b in range(B):
                        all_bits_list = bits_batch[b]
                        if all_bits_list:
                            t = torch.tensor(all_bits_list, device=device, dtype=torch.float32)
                            pad = (6 - len(all_bits_list) % 6) % 6
                            if pad > 0:
                                t = torch.cat([t, torch.zeros(pad, device=device)])
                            bits_tensors.append(t)
                            max_len = max(max_len, t.shape[0])
                        else:
                            bits_tensors.append(torch.zeros(6, device=device, dtype=torch.float32))
                            max_len = max(max_len, 6)
                    if bits_tensors:
                        pad_to = ((max_len + 5) // 6) * 6
                        stacked = []
                        for t in bits_tensors:
                            if t.shape[0] < pad_to:
                                t = torch.cat([t, torch.zeros(pad_to - t.shape[0], device=device)])
                            stacked.append(t)
                        bits_batch_tensor = torch.stack(stacked)
                        symbols_batch = ibp_mod.modulate(bits_batch_tensor, snr_db=snr_db)
                        demod_batch = ibp_mod.demodulate(symbols_batch)
                elif mode == 'qam':
                    bits_tensors = []
                    max_len = 0
                    for b in range(B):
                        all_bits_list = bits_batch[b]
                        if all_bits_list:
                            t = torch.tensor(all_bits_list, device=device, dtype=torch.float32)
                            pad = (6 - len(all_bits_list) % 6) % 6
                            if pad > 0:
                                t = torch.cat([t, torch.zeros(pad, device=device)])
                            bits_tensors.append(t)
                            max_len = max(max_len, t.shape[0])
                        else:
                            bits_tensors.append(torch.zeros(6, device=device, dtype=torch.float32))
                            max_len = max(max_len, 6)
                    if bits_tensors:
                        pad_to = ((max_len + 5) // 6) * 6
                        stacked = []
                        for t in bits_tensors:
                            if t.shape[0] < pad_to:
                                t = torch.cat([t, torch.zeros(pad_to - t.shape[0], device=device)])
                            stacked.append(t)
                        bits_batch_tensor = torch.stack(stacked)
                        symbols_batch = qam_mod.modulate(bits_batch_tensor)
                        noise_power = _noise_power_from_symbols(symbols_batch, snr_db)
                        noise_real = torch.randn_like(symbols_batch.real, device=device) * math.sqrt(noise_power / 2)
                        noise_imag = torch.randn_like(symbols_batch.imag, device=device) * math.sqrt(noise_power / 2)
                        noisy_symbols = symbols_batch + torch.complex(noise_real, noise_imag)
                        demod_batch_qam = qam_mod.demodulate(noisy_symbols)
                for b in range(B):
                    if total_per_snr[snr_db] >= num:
                        break
                    feats = batch_feats[b]
                    bit_allocation = bit_alloc_batch[b]
                    all_bits_list = bits_batch[b]
                    image_bits_info = per_image_bits_batch[b]
                    image_total_bits = image_bits_info['total']
                    image_imp_bits = image_bits_info.get('imp', 0)
                    image_nsk_bits = image_bits_info.get('nsk', 0)

                    if all_bits_list:
                        if mode == 'ibp':
                            demod_bits = demod_batch[b:b + 1]
                            rec = torch.zeros_like(feats)
                            unpacked = unpack_ibp_dual_stream(demod_bits, bit_allocation, batch_idx=0,
                                                              device=feats.device)
                            for feat_idx, int_val, total_bits in unpacked:
                                rec[feat_idx] = dequantize_from_int(int_val, total_bits, device_=feats.device).squeeze()
                        else:
                            demod_bits = demod_batch_qam[b:b + 1]
                            rec = torch.zeros_like(feats)
                            actual_data_bits = sum(alloc[1] for alloc in bit_allocation)
                            bit_idx = 0
                            for alloc in bit_allocation:
                                feat_idx = alloc[0]
                                total_bits = alloc[1]
                                if bit_idx >= actual_data_bits:
                                    break
                                if bit_idx + total_bits <= min(demod_bits.shape[1], actual_data_bits):
                                    feature_bits = demod_bits[0, bit_idx:bit_idx + total_bits]
                                    int_val = 0
                                    for j in range(min(total_bits, len(feature_bits))):
                                        if feature_bits[j] > 0.5:
                                            int_val += (1 << j)
                                    rec[feat_idx] = dequantize_from_int(int_val, total_bits,
                                                                        device_=feats.device).squeeze()
                                    bit_idx += total_bits
                                else:
                                    break
                    else:
                        rec = torch.zeros_like(feats)
                    if mode == 'ibp':
                        per_image_bits_by_snr[snr_db].append(
                            {'total': image_total_bits, 'imp': image_imp_bits, 'nsk': image_nsk_bits})
                    else:
                        per_image_bits_by_snr[snr_db].append({'total': image_total_bits})
                    logits = decoder(rec.unsqueeze(0))
                    pred = logits.argmax(dim=1).item()
                    correct_per_snr[snr_db] += int(pred == labels[b].item())
                    total_per_snr[snr_db] += 1

    for snr_db in snr_values:
        acc = 100.0 * correct_per_snr[snr_db] / max(1, total_per_snr[snr_db])
        accuracies.append(acc)
        snr_str = f"{snr_db}dB" if snr_db is not None else "无噪声"
        print(f"  SNR={snr_str}: {acc:.2f}%")
    total_images_evaluated = total_per_snr[snr_values[0]] if snr_values else 0
    all_per_image_bits = per_image_bits_by_snr[snr_values[-1]] if snr_values else []

    # 打印比特使用统计（特征级别）
    print(f"\n{mode.upper()} 比特使用统计（特征级别）:")
    if mode == 'ibp':
        total_uses = sum(ibp_bit_stats.values())
        if total_uses > 0:
            total_bits_all = sum((imp + nsk) * cnt for (imp, nsk), cnt in ibp_bit_stats.items())
            n_im = max(1, total_images_evaluated)
            print(f"  总使用次数: {total_uses}  总比特: {total_bits_all}  评估图片数: {total_images_evaluated}")
            print(f"  每张图片平均使用比特: {total_bits_all / n_im:.2f}")
            for (imp, nsk), count in sorted(ibp_bit_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                pct = 100.0 * count / total_uses
                print(f"  (imp={imp}, nsk={nsk}): {count}次 ({pct:.1f}%)")
    else:
        total_uses = sum(qam_bit_stats.values())
        if total_uses > 0:
            total_bits_all = sum(bits * cnt for bits, cnt in qam_bit_stats.items())
            n_im = max(1, total_images_evaluated)
            print(f"  总使用次数: {total_uses}  总比特: {total_bits_all}  评估图片数: {total_images_evaluated}")
            print(f"  每张图片平均使用比特: {total_bits_all / n_im:.2f}")
            for bits, count in sorted(qam_bit_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                pct = 100.0 * count / total_uses
                print(f"  {bits}比特: {count}次 ({pct:.1f}%)")

    # 打印每个图片的比特使用统计（使用最后一个SNR的统计）
    if len(all_per_image_bits) > 0:
        print(f"\n{mode.upper()} 每个图片的比特使用统计 (SNR={snr_values[-1]}dB):")
        if mode == 'ibp':
            total_bits_list = [x['total'] for x in all_per_image_bits]
            imp_bits_list = [x['imp'] for x in all_per_image_bits]
            nsk_bits_list = [x['nsk'] for x in all_per_image_bits]

            print(f"  总比特数:")
            print(f"    平均: {np.mean(total_bits_list):.2f} 比特/图片")
            print(f"    最小: {np.min(total_bits_list)} 比特")
            print(f"    最大: {np.max(total_bits_list)} 比特")
            print(f"    标准差: {np.std(total_bits_list):.2f}")

            print(f"  重要比特数:")
            print(f"    平均: {np.mean(imp_bits_list):.2f} 比特/图片")
            print(f"    最小: {np.min(imp_bits_list)} 比特")
            print(f"    最大: {np.max(imp_bits_list)} 比特")
            print(f"    标准差: {np.std(imp_bits_list):.2f}")

            print(f"  非重要比特数:")
            print(f"    平均: {np.mean(nsk_bits_list):.2f} 比特/图片")
            print(f"    最小: {np.min(nsk_bits_list)} 比特")
            print(f"    最大: {np.max(nsk_bits_list)} 比特")
            print(f"    标准差: {np.std(nsk_bits_list):.2f}")

            print(f"  重要比特占比:")
            imp_ratios = [imp / max(total, 1) for imp, total in zip(imp_bits_list, total_bits_list)]
            print(f"    平均: {np.mean(imp_ratios) * 100:.2f}%")
            print(f"    最小: {np.min(imp_ratios) * 100:.2f}%")
            print(f"    最大: {np.max(imp_ratios) * 100:.2f}%")

            # 打印前10个图片的详细统计
            print(f"\n  前10个图片的详细统计:")
            for idx in range(min(10, len(all_per_image_bits))):
                bits_info = all_per_image_bits[idx]
                imp_ratio = bits_info['imp'] / max(bits_info['total'], 1) * 100
                print(f"    图片{idx + 1}: 总={bits_info['total']:4d}, "
                      f"重要={bits_info['imp']:4d}, 非重要={bits_info['nsk']:4d}, "
                      f"重要占比={imp_ratio:5.2f}%")
        else:
            # QAM模式
            total_bits_list = [x['total'] for x in all_per_image_bits]

            print(f"  总比特数:")
            print(f"    平均: {np.mean(total_bits_list):.2f} 比特/图片")
            print(f"    最小: {np.min(total_bits_list)} 比特")
            print(f"    最大: {np.max(total_bits_list)} 比特")
            print(f"    标准差: {np.std(total_bits_list):.2f}")

            # 打印前10个图片的详细统计
            print(f"\n  前10个图片的详细统计:")
            for idx in range(min(10, len(all_per_image_bits))):
                bits_info = all_per_image_bits[idx]
                print(f"    图片{idx + 1}: {bits_info['total']:4d} 比特")

    return accuracies


def evaluate_som_mode(encoder, decoder, omega_per_image_test, test_loader, L=2, M=4, max_symbols=166, num=1000,
                      snr_values=None,
                      use_per_image_omega_eval: bool = True, features_per_image_test=None):
    # 确保所有模型处于评估模式
    encoder.eval()
    decoder.eval()
    """
    评估SOM模式的性能 - 直接将特征映射到符号，但遵循资源约束。
    test_loader 须返回 (imgs, labels, indices)。语义重要性由 use_per_image_omega_eval 控制：
    - True: 每张图使用 omega_per_image_test[indices[b]]（每图重要性）
    - False: 所有图使用全局重要性 omega_per_image_test.mean(dim=0)

    Args:
        encoder: 语义编码器（与QAM/IBP相同）
        decoder: 语义解码器（与QAM/IBP相同）
        omega_per_image_test: 每图语义重要性 [N_test, 512]
        test_loader: 须为 DatasetWithIndex 的 loader，返回 (imgs, labels, indices)
        L: SOM层数（默认3）
        M: SOM每层调制阶数（默认4）
        max_symbols: 最大符号数（默认166，对应1000比特预算，1000/6≈166）
        use_per_image_omega_eval: 若 True 按每图重要性选特征，若 False 按全局重要性选特征（与 QAM/IBP 评估一致）

    注意：
    - 使用相同的encoder/decoder（✓）
    - 遵循资源约束：最多传输max_symbols个符号（每对特征需要L层符号）
    - 按 use_per_image_omega_eval 选择全局/每图重要性，再按重要性选特征两两成对送入 SOM
    - 每个特征对需要L个符号（L层），所以最多可传输 max_symbols//L 对特征
    """
    som_mod = SOMModulator(L=L, M=M, gamma=1.0)
    use_per_img = use_per_image_omega_eval
    omega_eval_global = None if use_per_img else omega_per_image_test.mean(dim=0).to(device)

    accuracies = []

    print(f"\n{'=' * 70}")
    print(f"【SOM模式】评估开始 (L={L}, M={M}, 最大符号数={max_symbols})...")
    print(f"  语义重要性: {'每图' if use_per_img else '全局'}")
    print(f"  资源约束: 最多{max_symbols}个符号 (对应约{max_symbols * 6}比特预算)")
    print(f"  每个特征对需要{L}个符号，最多传输{max_symbols // L}对特征")
    print(f"{'=' * 70}")

    for snr_db in snr_values:
        correct = total = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"SOM SNR={snr_db}dB",
                              total=min(20, len(test_loader))):
                if total >= num:
                    break
                if len(batch) != 3:
                    raise ValueError(
                        "evaluate_som_mode 需要 test_loader 返回 (imgs, labels, indices)，请使用 DatasetWithIndex。")
                images, labels, indices = batch[0], batch[1], batch[2]
                if torch.is_tensor(indices):
                    indices = indices.cpu().tolist()
                images, labels = images.to(device), labels.to(device)
                for b in range(images.size(0)):
                    if total >= num:
                        break

                    # 提取特征（与QAM/IBP相同）
                    if features_per_image_test is not None:
                        feats = features_per_image_test[indices[b]].to(device)
                    else:
                        feats = encoder(images[b:b + 1])[0]  # [512]

                    # 按 use_per_image_omega_eval 选择全局或每图重要性，再选要传输的特征
                    if use_per_img:
                        om_b = omega_per_image_test[indices[b]].to(feats.device)
                    else:
                        om_b = omega_eval_global.to(feats.device)
                    max_pairs = max_symbols // L  # 最多可传输的特征对数
                    n_keep = min(2 * max_pairs, feats.shape[0])  # 最多 2*max_pairs 个单维
                    top_indices = torch.argsort(om_b, descending=True)[:n_keep]  # [n_keep]

                    if n_keep == 0:
                        rec_features = torch.zeros_like(feats).unsqueeze(0)
                    else:
                        # 按单维重要性顺序取出特征，再两两成对送入 SOM
                        selected_feats_tensor = feats[top_indices].unsqueeze(0)  # [1, n_keep]

                        # SOM调制（添加噪声）
                        symbols = som_mod.modulate(selected_feats_tensor, snr_db=snr_db)  # [1, N_pairs, L]

                        # SOM解调
                        rec_selected_feats = som_mod.demodulate(symbols)  # [1, n_keep]

                        # 重建完整特征向量：按单维索引填回
                        rec_features = torch.zeros_like(feats).unsqueeze(0)  # [1, 512]
                        rec_features[0, top_indices] = rec_selected_feats[0]

                    # 分类（使用相同的decoder）
                    logits = decoder(rec_features)
                    pred = logits.argmax(dim=1).item()
                    correct += int(pred == labels[b].item())
                    total += 1

        acc = 100.0 * correct / max(1, total)
        accuracies.append(acc)
        print(f"  SNR={snr_db:3d}dB: {acc:.2f}%")

    print(f"\nSOM评估完成")
    print(f"  实际传输: 最多{max_symbols // L}对特征 ({max_symbols // L * 2}个特征), {max_symbols}个符号")
    return accuracies, snr_values


def evaluate_sdmcm_mode(encoder, decoder, omega_per_image_test, test_loader, n_bits=4, m_bits=6, max_symbols=166,
                        num=1000,
                        snr_values=None, use_per_image_omega_eval: bool = True, features_per_image_test=None):
    # 确保所有模型处于评估模式
    encoder.eval()
    decoder.eval()
    """
    评估 sDMCM 模式的性能 - 直接将特征映射到符号，遵循资源约束。
    test_loader 须返回 (imgs, labels, indices)。语义重要性由 use_per_image_omega_eval 控制：
    - True: 每张图使用 omega_per_image_test[indices[b]]（每图重要性）
    - False: 所有图使用全局重要性 omega_per_image_test.mean(dim=0)

    Args:
        encoder: 语义编码器（与QAM/IBP相同）
        decoder: 语义解码器（与QAM/IBP相同）
        omega_per_image_test: 每图语义重要性 [N_test, 512]
        test_loader: 须为 DatasetWithIndex 的 loader，返回 (imgs, labels, indices)
        n_bits: 量化比特数（默认4）
        m_bits: 调制阶数（默认6，即64-QAM）
        max_symbols: 最大符号数（默认166，对应1000比特预算，1000/6≈166）
        use_per_image_omega_eval: 若 True 按每图重要性选特征，若 False 按全局重要性选特征（与 QAM/IBP 评估一致）

    注意：
    - 使用相同的encoder/decoder（✓）
    - 遵循资源约束：最多传输max_symbols个符号
    - 按 use_per_image_omega_eval 选择全局/每图重要性，再选特征两两成对送入 sDMCM
    - 每个符号对应2个特征（I和Q分量）
    """
    sdmcm_mod = SDMCMMapper(n_bits=n_bits, m_bits=m_bits, gamma=1.0)
    use_per_img = use_per_image_omega_eval
    omega_eval_global = None if use_per_img else omega_per_image_test.mean(dim=0).to(device)

    accuracies = []

    print(f"\n{'=' * 70}")
    print(f"【sDMCM模式】评估开始 (n={n_bits}, m={m_bits}, 最大符号数={max_symbols})...")
    print(f"  语义重要性: {'每图' if use_per_img else '全局'}")
    print(f"  资源约束: 最多{max_symbols}个符号 (对应约{max_symbols * m_bits}比特预算)")
    print(f"  每个符号对应2个特征，最多传输{max_symbols * 2}个特征")
    print(f"{'=' * 70}")

    for snr_db in snr_values:
        correct = total = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"sDMCM SNR={snr_db}dB",
                              total=min(20, len(test_loader))):
                if total >= num:
                    break
                if len(batch) != 3:
                    raise ValueError(
                        "evaluate_sdmcm_mode 需要 test_loader 返回 (imgs, labels, indices)，请使用 DatasetWithIndex。")
                images, labels, indices = batch[0], batch[1], batch[2]
                if torch.is_tensor(indices):
                    indices = indices.cpu().tolist()
                images, labels = images.to(device), labels.to(device)
                for b in range(images.size(0)):
                    if total >= num:
                        break

                    # 提取特征（与QAM/IBP相同）
                    if features_per_image_test is not None:
                        feats = features_per_image_test[indices[b]].to(device)
                    else:
                        feats = encoder(images[b:b + 1])[0]  # [512]

                    # 按 use_per_image_omega_eval 选择全局或每图重要性，再选要传输的特征
                    if use_per_img:
                        om_b = omega_per_image_test[indices[b]].to(feats.device)
                    else:
                        om_b = omega_eval_global.to(feats.device)
                    n_keep = min(2 * max_symbols, feats.shape[0])  # 每符号 2 维，共 max_symbols 个符号
                    top_indices = torch.argsort(om_b, descending=True)[:n_keep]  # [n_keep]

                    if n_keep == 0:
                        rec_features = torch.zeros_like(feats).unsqueeze(0)
                    else:
                        # 按单维重要性顺序取出特征，再两两成对送入 sDMCM
                        selected_feats_tensor = feats[top_indices].unsqueeze(0)  # [1, n_keep]

                        # sDMCM调制（添加噪声）
                        symbols = sdmcm_mod.modulate(selected_feats_tensor, snr_db=snr_db)  # [1, N_symbols]

                        # sDMCM解调
                        rec_selected_feats = sdmcm_mod.demodulate(symbols)  # [1, n_keep]

                        # 重建完整特征向量：按单维索引填回
                        rec_features = torch.zeros_like(feats).unsqueeze(0)  # [1, 512]
                        rec_features[0, top_indices] = rec_selected_feats[0]

                    # 分类（使用相同的decoder）
                    logits = decoder(rec_features)
                    pred = logits.argmax(dim=1).item()
                    correct += int(pred == labels[b].item())
                    total += 1

        acc = 100.0 * correct / max(1, total)
        accuracies.append(acc)
        print(f"  SNR={snr_db:3d}dB: {acc:.2f}%")

    print(f"\nsDMCM评估完成")
    print(f"  实际传输: 最多{max_symbols}个符号 ({max_symbols * 2}个特征)")
    return accuracies, snr_values


def train_jscc_mode(train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                    epochs: int = 50, snr_train_db: float = 4.0,
                    snr_schedule: Optional[List[float]] = None,
                    save_paths: Tuple[str, str] = ("jscc_encoder.pth", "jscc_decoder.pth")):
    """
    训练 JSCC：仅一对编解码器，对接处166符号。
    - JSCCEncoder(图像) → 166符号 → AWGN → JSCCDecoder → 10类
    - 不依赖语义 encoder/decoder，完全独立
    """
    enc_path, dec_path = save_paths
    enc_abs = os.path.abspath(enc_path)
    dec_abs = os.path.abspath(dec_path)
    if os.path.exists(enc_abs) and os.path.exists(dec_abs):
        enc = JSCCEncoder().to(device)
        dec = JSCCDecoder().to(device)
        enc.load_state_dict(torch.load(enc_abs, map_location=device, weights_only=True))
        dec.load_state_dict(torch.load(dec_abs, map_location=device, weights_only=True))
        print(f"  ✓ 发现JSCC模型 ({enc_path}, {dec_path})，跳过训练")
        return enc, dec

    enc = JSCCEncoder().to(device)
    dec = JSCCDecoder().to(device)

    optim_all = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optim_all, T_max=epochs, eta_min=1e-6)
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0

    print(f"\n{'=' * 70}")
    print(f"【JSCC模式】一对编解码器，对接166符号...")
    print(f"{'=' * 70}")

    for ep in tqdm(range(epochs), desc="JSCC train"):
        enc.train()
        dec.train()
        for batch in train_loader:
            if len(batch) != 3:
                imgs, labels = batch[0], batch[1]
            else:
                imgs, labels = batch[0], batch[1]
            imgs, labels = imgs.to(device), labels.to(device)

            symbols = enc(imgs)
            snr_db = snr_train_db
            if snr_schedule is not None and len(snr_schedule) > 0:
                snr_db = float(np.random.choice(snr_schedule))
            noisy = _jscc_awgn_complex(symbols, snr_db)
            logits = dec(noisy)

            loss = ce(logits, labels)
            optim_all.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
            optim_all.step()

        scheduler.step()

        enc.eval()
        dec.eval()
        correct = total = 0
        val_loader_use = val_loader if val_loader is not None else train_loader
        max_val_batches = 20 if val_loader is None else 9999
        with torch.no_grad():
            for vi, batch in enumerate(val_loader_use):
                if vi >= max_val_batches:
                    break
                if len(batch) != 3:
                    imgs, labels = batch[0], batch[1]
                else:
                    imgs, labels = batch[0], batch[1]
                imgs, labels = imgs.to(device), labels.to(device)
                symbols = enc(imgs)
                logits = dec(symbols)
                pred = logits.argmax(dim=1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        acc = 100.0 * correct / max(1, total)
        if acc > best_acc:
            best_acc = acc
            torch.save(enc.state_dict(), enc_path)
            torch.save(dec.state_dict(), dec_path)

    print(f"  JSCC 训练完成，最佳验证准确率: {best_acc:.2f}%")
    return enc, dec


def evaluate_jscc_mode(jscc_encoder: nn.Module, jscc_decoder: nn.Module, test_loader: DataLoader,
                       num: int = 1000, snr_values: Optional[List[float]] = None):
    """
    评估 JSCC 模式：一对编解码器，对接166符号，在 -6 到 10 dB（步长2）下测试任务成功率。
    """
    jscc_encoder.eval()
    jscc_decoder.eval()

    if snr_values is None:
        snr_values = list(range(-6, 11, 2))

    n_symbols = JSCC_N_SYMBOLS

    accuracies = []
    print(f"\n{'=' * 70}")
    print(f"【JSCC模式】评估开始 (一对模型, 对接{n_symbols}符号)...")
    print(f"{'=' * 70}")

    for snr_db in snr_values:
        correct = total = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"JSCC SNR={snr_db}dB", total=min(20, len(test_loader))):
                if total >= num:
                    break
                images, labels = batch[0], batch[1]
                images, labels = images.to(device), labels.to(device)

                symbols = jscc_encoder(images)
                noisy = _jscc_awgn_complex(symbols, snr_db)
                logits = jscc_decoder(noisy)

                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        acc = 100.0 * correct / max(1, total)
        accuracies.append(acc)
        print(f"  SNR={snr_db:3d}dB: {acc:.2f}%")

    print(f"\nJSCC评估完成")
    print(f"  对接: {n_symbols}复符号 (≈996比特)")
    return accuracies, snr_values


def main():
    import matplotlib.pyplot as plt

    set_seed(42)

    print("=" * 70)
    print("完整训练和评估脚本 - 支持QAM、IBP、SOM、sDMCM和JSCC五种方案")
    print("=" * 70)
    if torch.cuda.is_available():
        print(
            f"  当前使用设备: {device}  (双GPU时仅用5080请先设置: PowerShell: $env:CUDA_VISIBLE_DEVICES='1'; CMD: set CUDA_VISIBLE_DEVICES=1)")
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")  # 消除 TF32 警告并可能提升 matmul 性能

    # ========== 步骤1: 数据集检查 ==========
    print("\n【步骤1】检查数据集...")
    # DPPO 在线编码时每次 __getitem__ 随机裁剪，同一图像索引在不同 epoch 呈现不同语义特征，等效扩大训练分布
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.64, 1.0), ratio=(0.75, 1.333)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_set = datasets.STL10(root='./data', split='train', download=True, transform=tf_train)
    # 减少num_workers和禁用persistent_workers以节省内存
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    tf_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_set = datasets.STL10(root='./data', split='test', download=True, transform=tf_test)
    print("✓ 训练集与测试集已准备")

    # ========== 步骤2: 语义编解码器检查 ==========
    print("\n【步骤2】检查语义编解码器...")
    encoder = SemanticEncoder(512).to(device)
    decoder = SemanticDecoder(512, 10).to(device)
    if os.path.exists("best_encoder.pth"):
        print("✓ 加载已保存的编解码器")
        encoder.load_state_dict(torch.load("best_encoder.pth", map_location=device))
        decoder.load_state_dict(torch.load("best_decoder.pth", map_location=device))
    else:
        print("✗ 训练编解码器（联合训练 encoder+decoder）...")
        val_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2, persistent_workers=False)
        pretrain_encoder_decoder(encoder, decoder, train_loader, val_loader, joint_only=True)
    encoder.eval();
    decoder.eval()
    # PyTorch 2.0+ 可选编译加速（需 Triton，Windows 上通常未安装，故先检查再编译，避免首次前向时报 TritonMissing）
    _use_compile = False
    if getattr(torch, "compile", None) is not None and device.type == "cuda":
        try:
            import triton  # noqa: F401
            _use_compile = True
        except ImportError:
            pass
    if _use_compile:
        try:
            encoder = torch.compile(encoder, mode="reduce-overhead")
            decoder = torch.compile(decoder, mode="reduce-overhead")
            print("  [OK] encoder/decoder 已使用 torch.compile 加速")
        except Exception as e:
            print(f"  [INFO] torch.compile 未启用: {e}")

    # ========== 步骤3: 语义重要性一次性预计算（训练/评估时仅按索引取用，无重复计算）==========
    # 两种用法：① 每图重要性 omega_per_image[indices]（use_per_image_omega_*=True）
    #          ② 全局重要性 omega = omega_per_image_train.mean(0)（use_per_image_omega_*=False）
    # 均在此步算好或从文件加载，后续训练与评估不再调用 compute_omega / compute_omega_per_image。
    print("\n【步骤3】检查每图语义重要性（训练集+测试集）...")
    OMEGA_PER_IMAGE_PATH = "omega_per_image.pth"
    need_recompute = False
    if os.path.exists(OMEGA_PER_IMAGE_PATH):
        data = torch.load(OMEGA_PER_IMAGE_PATH, map_location="cpu")
        if isinstance(data, dict) and "train" in data and "test" in data:
            omega_per_image_train = data["train"]
            omega_per_image_test = data["test"]
            n_train_saved = data.get("n_train", omega_per_image_train.shape[0])
            n_test_saved = data.get("n_test", omega_per_image_test.shape[0])
            if n_train_saved != len(train_set) or n_test_saved != len(test_set):
                need_recompute = True
                print(
                    f"  [警告] 保存的样本数与当前数据集不一致（train {n_train_saved} vs {len(train_set)}, test {n_test_saved} vs {len(test_set)}），将重新计算。")
            else:
                print("✓ 加载已保存的每图语义重要性: " + OMEGA_PER_IMAGE_PATH + "（训练集+测试集）")
        else:
            need_recompute = True
    else:
        need_recompute = True

    if need_recompute:
        print("✗ 预计算训练集每图语义重要性...")
        omega_per_image_train = compute_omega_per_image(encoder, decoder, train_set, batch_size=64)
        print("✗ 预计算测试集每图语义重要性...")
        omega_per_image_test = compute_omega_per_image(encoder, decoder, test_set, batch_size=64)
        torch.save({
            "train": omega_per_image_train,
            "n_train": len(train_set),
            "test": omega_per_image_test,
            "n_test": len(test_set),
        }, OMEGA_PER_IMAGE_PATH)
        print("✓ 已保存: " + OMEGA_PER_IMAGE_PATH)

    omega_per_image_train = omega_per_image_train.to(device)
    omega_per_image_test = omega_per_image_test.to(device)
    omega = omega_per_image_train.mean(dim=0).to(device)

    print("\n" + "=" * 70)
    print(
        "语义重要性（每图）: 训练集 " + str(omega_per_image_train.shape) + ", 测试集 " + str(omega_per_image_test.shape))
    print("  全局均值 omega 仅用于兼容；训练/评估时均按样本索引使用各自的 omega。")
    print("=" * 70)

    # ========== 步骤3b: 语义特征预计算（训练/评估时按索引取用，避免重复调用 encoder）==========
    FEATURES_PER_IMAGE_PATH = "features_per_image.pth"
    need_recompute_feats = False
    features_per_image_train = None
    features_per_image_test = None
    if os.path.exists(FEATURES_PER_IMAGE_PATH):
        try:
            data = torch.load(FEATURES_PER_IMAGE_PATH, map_location="cpu")
            if isinstance(data, dict) and "train" in data and "test" in data:
                features_per_image_train = data["train"]
                features_per_image_test = data["test"]
                n_train_f = data.get("n_train", features_per_image_train.shape[0])
                n_test_f = data.get("n_test", features_per_image_test.shape[0])
                if n_train_f != len(train_set) or n_test_f != len(test_set):
                    need_recompute_feats = True
                    print(f"  [警告] 保存的特征数与当前数据集不一致，将重新计算。")
                else:
                    print("✓ 加载已保存的语义特征: " + FEATURES_PER_IMAGE_PATH + "（训练集+测试集）")
            else:
                need_recompute_feats = True
        except Exception:
            need_recompute_feats = True
    else:
        need_recompute_feats = True

    if need_recompute_feats:
        print("✗ 预计算训练集语义特征...")
        features_per_image_train = compute_features_per_image(encoder, train_set, batch_size=64)
        print("✗ 预计算测试集语义特征...")
        features_per_image_test = compute_features_per_image(encoder, test_set, batch_size=64)
        torch.save({
            "train": features_per_image_train,
            "n_train": len(train_set),
            "test": features_per_image_test,
            "n_test": len(test_set),
        }, FEATURES_PER_IMAGE_PATH)
        print("✓ 已保存: " + FEATURES_PER_IMAGE_PATH)

    if features_per_image_train is not None:
        features_per_image_train = features_per_image_train.to(device)
    if features_per_image_test is not None:
        features_per_image_test = features_per_image_test.to(device)

    train_set_indexed = DatasetWithIndex(train_set)
    LOW_MEMORY = True  # 内存紧张时减少 workers 和 batch，降低 CPU/GPU 占用
    nw = 2 if LOW_MEMORY else 8
    train_loader_indexed = DataLoader(
        train_set_indexed, batch_size=128, shuffle=True, num_workers=nw, pin_memory=True, persistent_workers=(nw > 0),
        prefetch_factor=2
    )
    test_set_indexed = DatasetWithIndex(test_set)
    test_loader_indexed = DataLoader(
        test_set_indexed, batch_size=32, shuffle=False, num_workers=2, persistent_workers=False,
        pin_memory=(device.type == "cuda"), prefetch_factor=2 if device.type == "cuda" else None
    )

    # ========== 步骤4: 训练QAM和IBP两种DPPO模型 ==========
    print("\n【步骤4】训练两种DPPO模型（固定SNR=5dB）...")
    # 每特征允许分配的最大比特数（动作上限）：训练与评估必须一致
    ACTION_MAX_PER_STEP = 8
    # False：DPPO 每步对增强后的图像在线 encoder，语义状态随 RandomResizedCrop 等变化；True：用上方预计算特征（更快、易过拟合）
    USE_CACHED_FEATURES_FOR_DPPO = False
    if not USE_CACHED_FEATURES_FOR_DPPO:
        print("  DPPO 训练：在线 encoder + 训练集增强（RandomResizedCrop 等），不使用 features 缓存。")
    else:
        print("  DPPO 训练：使用预计算 train 特征缓存。")

    # 恢复训练配置
    RESUME_FROM_10DB = 1  # 0=有 ppo_actor/critic 则只加载并跳过训练；1=加载后继续训练
    ADDITIONAL_EPOCHS = 30 # resume=1 时作为继续训练的 epoch 数（>0）；若为 0 则继续训练仍用上面的 epochs=
    # 课程学习：从 PPOConfig 读取，True 时 IBP/QAM 均在 -6~4dB 均分训练
    _train_cfg = PPOConfig(use_curriculum=False)
    USE_CURRICULUM = _train_cfg.use_curriculum
    '''
    # 训练QAM模式
    print("\n--- 训练QAM模式 ---")
    ppo_qam, rewards_qam = train_dppo_mode(
        encoder, decoder, omega, test_loader_indexed,
        mode='qam',
        epochs=13,
        iterations_per_epoch=10,
        use_curriculum=USE_CURRICULUM,
        resume_from_10db=RESUME_FROM_10DB,
        additional_epochs=ADDITIONAL_EPOCHS,
        omega_per_image=omega_per_image_test,
        features_per_image=(features_per_image_test if USE_CACHED_FEATURES_FOR_DPPO else None),
        low_memory_mode=LOW_MEMORY,
    )
    '''
    # 训练IBP模式
    print("\n--- 训练IBP模式 ---")
    ppo_ibp, rewards_ibp = train_dppo_mode(
        encoder, decoder, omega, test_loader_indexed,
        mode='ibp',
        epochs=72,
        iterations_per_epoch=10,
        max_bits=ACTION_MAX_PER_STEP,
        use_curriculum=USE_CURRICULUM,
        resume_from_10db=RESUME_FROM_10DB,
        additional_epochs=ADDITIONAL_EPOCHS,
        omega_per_image=omega_per_image_test,
        features_per_image=(features_per_image_test if USE_CACHED_FEATURES_FOR_DPPO else None),
        low_memory_mode=LOW_MEMORY,
    )

    # ========== 步骤4b: 训练JSCC基准方案 ==========
    print("\n--- 训练JSCC模式（基准方案）---")
    val_loader_jscc = DataLoader(
        test_set_indexed, batch_size=64, shuffle=False, num_workers=2, persistent_workers=False,
        pin_memory=(device.type == "cuda")
    )
    jscc_enc, jscc_dec = train_jscc_mode(
        train_loader_indexed,
        val_loader=val_loader_jscc,
        epochs=100,
        snr_train_db=5.0,
        snr_schedule=list(range(-6, 11, 2)),
    )

    # ========== 步骤6: 评估五种模式 ==========
    print("\n【步骤6】评估四种模式...")
    # 评估时语义重要性：True=每图重要性，False=全局重要性（与 QAM/IBP/SOM/sDMCM 一致）
    use_per_image_omega_eval = True
    print(f"  评估语义重要性: {'每图' if use_per_image_omega_eval else '全局'}")
    print("\n对比说明:")
    print("  - 所有方案使用相同的语义编解码器（encoder/decoder）")
    print("  - 所有方案遵循相同的资源约束：1000比特预算 ≈ 166个符号")
    print("  - QAM/IBP: 使用DPPO动态分配比特给不同特征")
    print("  - TIBP: 前333特征中前111以3比特重要比特、后222以3比特非重要比特，IBP传输")
    print("  - SOM: 根据重要性权重选择特征对，每个特征对需要L个符号")
    print("  - sDMCM: 根据重要性权重选择特征对，使用语义数字调制星座映射")
    print("  - JSCC: 一对编解码器，对接166符号，端到端学习")
    print()

    snr_values1 = list(range(-6, 11, 2))  # 无噪声 1000dB 评估

    print(f"\n评估QAM模式:")
    acc_qam = evaluate_mode(encoder, decoder, omega_per_image_test, test_loader_indexed, ppo_qam, mode='qam', num=1000,
                            snr_values=snr_values1, use_per_image_omega_eval=use_per_image_omega_eval)

    print(f"\n评估IBP模式:")
    acc_ibp = evaluate_mode(encoder, decoder, omega_per_image_test, test_loader_indexed, ppo_ibp, mode='ibp', num=1000,
                            max_bits=ACTION_MAX_PER_STEP,
                            snr_values=snr_values1, use_per_image_omega_eval=use_per_image_omega_eval,
                            features_per_image_test=features_per_image_test)

    print(f"\n评估SOM模式:")
    # SOM使用相同的资源约束：1000比特预算 ≈ 166个符号（1000/6）
    # 每个特征对需要L=2个符号，所以最多传输166//2≈83对特征（166个特征）
    acc_som, snr_values_som = evaluate_som_mode(encoder, decoder, omega_per_image_test, test_loader_indexed,
                                                L=2, M=64, max_symbols=166, num=1000, snr_values=snr_values1,
                                                use_per_image_omega_eval=use_per_image_omega_eval,
                                                features_per_image_test=features_per_image_test)

 
    print(f"\n评估sDMCM模式:")
    # sDMCM使用相同的资源约束：1000比特预算 ≈ 166个符号（1000/6）
    # 每个符号对应2个特征，所以最多传输166*2=332个特征
    acc_sdmcm, snr_values_sdmcm = evaluate_sdmcm_mode(encoder, decoder, omega_per_image_test, test_loader_indexed,
                                                      n_bits=3, m_bits=3, max_symbols=166, num=100,
                                                      snr_values=snr_values1,
                                                      use_per_image_omega_eval=use_per_image_omega_eval,
                                                      features_per_image_test=features_per_image_test)
    
    print(f"\n评估JSCC模式（基准方案）:")
    if 'jscc_enc' not in locals() or 'jscc_dec' not in locals():
        jscc_enc = JSCCEncoder().to(device)
        jscc_dec = JSCCDecoder().to(device)
        if os.path.exists("jscc_encoder.pth") and os.path.exists("jscc_decoder.pth"):
            jscc_enc.load_state_dict(torch.load("jscc_encoder.pth", map_location=device))
            jscc_dec.load_state_dict(torch.load("jscc_decoder.pth", map_location=device))
            print("  [OK] 从文件加载JSCC模型")
        else:
            raise FileNotFoundError("未找到 jscc_encoder.pth / jscc_decoder.pth，请先运行JSCC训练")
    acc_jscc, snr_values_jscc = evaluate_jscc_mode(
        jscc_enc, jscc_dec, test_loader_indexed,
        num=1000, snr_values=snr_values1,
    )
    

if __name__ == "__main__":
    main()
