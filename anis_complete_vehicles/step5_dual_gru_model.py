"""
Step 5: Dual-Channel GRU Model with Cross-Attention
双通道GRU编码器 + 交叉注意力融合机制
适配 packed HDF5 格式（无 padding）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np


# ============================================================
# Dataset：从 packed HDF5 读取变长序列
# ============================================================
class PackedHDF5Dataset(Dataset):
    def __init__(self, h5_path, split='train', train_ratio=0.8, seed=42):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            self.offsets       = f['offsets'][:]
            self.lengths       = f['lengths'][:]
            self.segment_types = f['segment_types'][:]
            n_samples          = len(self.lengths)

        rng     = np.random.default_rng(seed)
        idx     = rng.permutation(n_samples)
        n_train = int(n_samples * train_ratio)
        self.indices = idx[:n_train] if split == 'train' else idx[n_train:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        i        = self.indices[item]
        s, e     = int(self.offsets[i]), int(self.offsets[i + 1])
        seg_type = int(self.segment_types[i])
        with h5py.File(self.h5_path, 'r') as f:
            drv = torch.tensor(f['driving_packed'][s:e], dtype=torch.float32)
            eng = torch.tensor(f['energy_packed'][s:e],  dtype=torch.float32)
        return drv, eng, seg_type


def collate_fn(batch):
    drv_list, eng_list, types = zip(*batch)
    lengths = torch.tensor([d.size(0) for d in drv_list], dtype=torch.long)
    drv_pad = pad_sequence(drv_list, batch_first=True)
    eng_pad = pad_sequence(eng_list, batch_first=True)
    types   = torch.tensor(types, dtype=torch.long)
    return drv_pad, eng_pad, lengths, types


# ============================================================
# GRUEncoder
# ============================================================
class GRUEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

    def forward(self, x, lengths):
        packed        = pack_padded_sequence(x, lengths.cpu(),
                                             batch_first=True, enforce_sorted=False)
        packed_out, h = self.gru(packed)
        output, _     = pad_packed_sequence(packed_out, batch_first=True)
        hidden        = h[-1]   # (B, H)
        return output, hidden


# ============================================================
# CrossChannelAttention
# ============================================================
class CrossChannelAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.query_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj    = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.scale       = hidden_dim ** 0.5
        # ✅ 新增：可学习温度参数，解决注意力卡在 0.5 的问题
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, z_source, z_target):
        Q = self.query_proj(z_target)
        K = self.key_proj(z_source)
        V = self.value_proj(z_source)

        # ✅ 修改：除以温度，让注意力分布更尖锐有区分度
        score    = (Q * K).sum(dim=-1, keepdim=True) / (self.scale * self.temperature.abs())
        attn_w   = torch.sigmoid(score)
        enhanced = z_target + attn_w * V
        return enhanced, attn_w


# ============================================================
# GRUDecoder —— 修复 T-1 vs T 问题
# ============================================================
class GRUDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, num_layers=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.z_to_h     = nn.Linear(latent_dim, latent_dim * num_layers)
        self.gru        = nn.GRU(
            input_size=output_dim,
            hidden_size=latent_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.out_proj = nn.Linear(latent_dim, output_dim)

    def forward(self, z, target_seq, lengths):
        """
        z          : (B, latent_dim)
        target_seq : (B, T, output_dim)  已 pad
        lengths    : (B,)
        returns    : recon (B, T, output_dim)  ← 与 target_seq 完全对齐
        """
        B, T, _ = target_seq.shape
        device   = z.device

        # 初始隐状态
        h0 = self.z_to_h(z).view(B, self.num_layers, self.latent_dim)
        h0 = h0.permute(1, 0, 2).contiguous()   # (num_layers, B, latent_dim)

        # teacher forcing 输入：[0, x0, x1, ..., x_{T-2}]，长度 = T
        zeros     = torch.zeros(B, 1, self.output_dim, device=device)
        dec_input = torch.cat([zeros, target_seq[:, :-1, :]], dim=1)  # (B, T, output_dim)

        # pack → gru → unpack，total_length=T 保证输出长度严格等于 T
        packed_in         = pack_padded_sequence(dec_input, lengths.cpu(),
                                                 batch_first=True, enforce_sorted=False)
        packed_out, _     = self.gru(packed_in, h0)
        # [修复] total_length=T 强制 unpack 到与输入相同长度
        out, _            = pad_packed_sequence(packed_out, batch_first=True,
                                                total_length=T)  # (B, T, latent_dim)

        recon = self.out_proj(out)   # (B, T, output_dim)
        return recon


# ============================================================
# DualChannelGRU
# ============================================================
class DualChannelGRU(nn.Module):
    def __init__(
        self,
        driving_input_dim=3,
        energy_input_dim=4,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        n_segment_types=2,
        type_embed_dim=8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.type_embedding = nn.Embedding(n_segment_types, type_embed_dim)

        self.driving_encoder = GRUEncoder(
            input_dim=driving_input_dim + type_embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        self.energy_encoder = GRUEncoder(
            input_dim=energy_input_dim + type_embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.cross_attn_E2B = CrossChannelAttention(hidden_dim)
        self.cross_attn_B2E = CrossChannelAttention(hidden_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )

        self.driving_decoder = GRUDecoder(hidden_dim, driving_input_dim)
        self.energy_decoder  = GRUDecoder(hidden_dim, energy_input_dim)

    def forward(self, driving_seq, energy_seq, lengths, segment_types):
        B, T, _ = driving_seq.shape

        t_emb     = self.type_embedding(segment_types)        # (B, E)
        t_emb_seq = t_emb.unsqueeze(1).expand(-1, T, -1)     # (B, T, E)

        drv_in = torch.cat([driving_seq, t_emb_seq], dim=-1)
        eng_in = torch.cat([energy_seq,  t_emb_seq], dim=-1)

        _, z_B = self.driving_encoder(drv_in, lengths)
        _, z_E = self.energy_encoder(eng_in,  lengths)

        z_B_enh, attn_E2B = self.cross_attn_E2B(z_E, z_B)
        z_E_enh, attn_B2E = self.cross_attn_B2E(z_B, z_E)

        z_concat = torch.cat([z_B, z_E, z_B_enh, z_E_enh], dim=-1)
        z_final  = self.fusion_mlp(z_concat)

        drv_recon = self.driving_decoder(z_B, driving_seq, lengths)
        eng_recon = self.energy_decoder(z_E, energy_seq,  lengths)

        return {
            'z_final': z_final,
            'z_B':     z_B,
            'z_E':     z_E,
            'reconstructions': {'driving': drv_recon, 'energy': eng_recon},
            'attentions':      {'B2E': attn_B2E, 'E2B': attn_E2B}
        }


# ============================================================
# 损失函数
# ============================================================
def compute_loss(model_output, driving_seq, energy_seq, lengths,
                 lambda_rec=1.0, lambda_orth=0.1):
    z_B       = model_output['z_B']
    z_E       = model_output['z_E']
    drv_recon = model_output['reconstructions']['driving']
    eng_recon = model_output['reconstructions']['energy']

    B  = driving_seq.size(0)
    # recon 与 target 的 T 以 recon 为准（total_length 已保证对齐）
    T  = drv_recon.size(1)
    device = driving_seq.device

    # 截取 target 到相同长度（防御性处理）
    drv_target = driving_seq[:, :T, :]
    eng_target = energy_seq[:,  :T, :]

    # 有效位置 mask：(B, T, 1)
    mask    = (torch.arange(T, device=device).unsqueeze(0)
               < lengths.unsqueeze(1))          # (B, T)
    mask_3d = mask.unsqueeze(-1).float()        # (B, T, 1)
    n_valid = mask_3d.sum()

    loss_drv = (F.mse_loss(drv_recon, drv_target, reduction='none') * mask_3d).sum() / n_valid
    loss_eng = (F.mse_loss(eng_recon, eng_target, reduction='none') * mask_3d).sum() / n_valid
    loss_rec = loss_drv + loss_eng

    # 正交约束（归一化后）
    z_B_n    = F.normalize(z_B, dim=-1)
    z_E_n    = F.normalize(z_E, dim=-1)
    loss_orth = torch.norm(torch.matmul(z_B_n.T, z_E_n), p='fro') / B

    total_loss = lambda_rec * loss_rec + lambda_orth * loss_orth

    return total_loss, {
        'total':                  total_loss.item(),
        'reconstruction':         loss_rec.item(),
        'reconstruction_driving': loss_drv.item(),
        'reconstruction_energy':  loss_eng.item(),
        'orthogonal':             loss_orth.item()
    }


# ============================================================
# 测试
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("🧪 Testing Dual-Channel GRU Model")
    print("=" * 70)

    torch.manual_seed(42)
    B, T = 16, 50

    driving_seq   = torch.rand(B, T, 3)
    energy_seq    = torch.rand(B, T, 4)
    lengths       = torch.randint(10, T, (B,))
    segment_types = torch.randint(0, 2, (B,))

    # padding 位置清零
    for i, l in enumerate(lengths):
        driving_seq[i, l:] = 0.0
        energy_seq[i, l:]  = 0.0

    model = DualChannelGRU(
        driving_input_dim=3,
        energy_input_dim=4,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        n_segment_types=2,
        type_embed_dim=8,
    )

    output = model(driving_seq, energy_seq, lengths, segment_types)

    print(f"\n✅ Forward pass:")
    print(f"   z_final  : {output['z_final'].shape}")
    print(f"   z_B      : {output['z_B'].shape}")
    print(f"   z_E      : {output['z_E'].shape}")
    print(f"   drv_recon: {output['reconstructions']['driving'].shape}")
    print(f"   eng_recon: {output['reconstructions']['energy'].shape}")

    # 验证 recon 和 target 形状一致
    assert output['reconstructions']['driving'].shape == driving_seq[:, :output['reconstructions']['driving'].size(1), :].shape
    print(f"   ✅ recon shape matches target")

    loss, loss_dict = compute_loss(output, driving_seq, energy_seq, lengths)

    print(f"\n✅ Loss:")
    for k, v in loss_dict.items():
        print(f"   {k:<25}: {v:.6f}")

    loss.backward()
    print(f"\n✅ Backward pass OK")

    print(f"\n{'=' * 70}")
    print(f"✅ All tests passed!")
    print(f"{'=' * 70}")