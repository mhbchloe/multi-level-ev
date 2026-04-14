"""
生成Transformer-AE完整架构图
展示数据从原始输入到聚类结果的完整流程
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np

fig = plt.figure(figsize=(20, 14))
ax = fig.add_subplot(111)
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# 颜色定义
color_input = '#E8F4F8'
color_encoder = '#FFE5B4'
color_latent = '#FFB6C1'
color_decoder = '#B4E7CE'
color_output = '#C8A2C8'
color_cluster = '#FFD700'

# ==================== 标题 ====================
ax.text(10, 13.5, 'Transformer-based Autoencoder Architecture\nfor EV Driving Behavior Clustering', 
        ha='center', va='top', fontsize=18, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='lightblue', alpha=0.9, edgecolor='black', linewidth=2))

# ==================== 1. 输入层 ====================
y_input = 11.5

# 输入数据块
input_box = FancyBboxPatch((0.5, y_input-0.6), 3, 1.5,
                           boxstyle="round,pad=0.15",
                           facecolor=color_input, edgecolor='black', linewidth=2)
ax.add_patch(input_box)

ax.text(2, y_input+0.6, 'Input Sequence', ha='center', va='center', 
        fontsize=12, fontweight='bold')
ax.text(2, y_input+0.2, 'Shape: (Batch, T, D)', ha='center', va='center', fontsize=9)
ax.text(2, y_input-0.1, 'T = 100 timesteps', ha='center', va='center', fontsize=8)
ax.text(2, y_input-0.35, 'D = 6 features:', ha='center', va='center', fontsize=8, fontweight='bold')
ax.text(2, y_input-0.55, '[Speed, Acc, SOC,\nVoltage, Current, Power]', 
        ha='center', va='center', fontsize=7, style='italic')

# 数据示例可视化（小热力图）
for i in range(5):
    for j in range(6):
        rect = Rectangle((0.7 + j*0.15, y_input-1.5 - i*0.12), 0.13, 0.1,
                        facecolor=plt.cm.viridis(np.random.rand()), 
                        edgecolor='white', linewidth=0.5)
        ax.add_patch(rect)

ax.text(2, y_input-1.8, 'Time →', ha='center', va='center', fontsize=7, style='italic')

# ==================== 2. Positional Encoding ====================
arrow1 = FancyArrowPatch((3.5, y_input), (4.5, y_input),
                        arrowstyle='->', mutation_scale=25, linewidth=3, color='black')
ax.add_patch(arrow1)

pos_box = FancyBboxPatch((4.5, y_input-0.4), 2, 0.8,
                         boxstyle="round,pad=0.1",
                         facecolor='#FFF8DC', edgecolor='black', linewidth=1.5)
ax.add_patch(pos_box)
ax.text(5.5, y_input+0.15, 'Positional Encoding', ha='center', va='center', 
        fontsize=10, fontweight='bold')
ax.text(5.5, y_input-0.15, 'PE(pos, 2i) = sin(pos/10000^(2i/d))', 
        ha='center', va='center', fontsize=7, family='monospace')

# ==================== 3. Transformer Encoder ====================
arrow2 = FancyArrowPatch((6.5, y_input), (7.5, y_input),
                        arrowstyle='->', mutation_scale=25, linewidth=3, color='black')
ax.add_patch(arrow2)

y_encoder = y_input

# Encoder主体
encoder_box = FancyBboxPatch((7.5, y_encoder-1.2), 4.5, 2.8,
                            boxstyle="round,pad=0.2",
                            facecolor=color_encoder, edgecolor='black', linewidth=3)
ax.add_patch(encoder_box)

ax.text(9.75, y_encoder+1.4, 'Transformer Encoder', ha='center', va='center', 
        fontsize=13, fontweight='bold')
ax.text(9.75, y_encoder+1.05, '(2 Layers)', ha='center', va='center', fontsize=9)

# Layer 1
layer1_y = y_encoder+0.4
# Multi-Head Attention
mha1 = FancyBboxPatch((8, layer1_y+0.2), 3.5, 0.5,
                      boxstyle="round,pad=0.05",
                      facecolor='#FFB347', edgecolor='black', linewidth=1.5)
ax.add_patch(mha1)
ax.text(9.75, layer1_y+0.45, 'Multi-Head Attention (4 heads)', 
        ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(9.75, layer1_y+0.25, 'd_model=64, d_k=16', 
        ha='center', va='center', fontsize=7)

# Feed Forward
ff1 = FancyBboxPatch((8, layer1_y-0.4), 3.5, 0.5,
                     boxstyle="round,pad=0.05",
                     facecolor='#77DD77', edgecolor='black', linewidth=1.5)
ax.add_patch(ff1)
ax.text(9.75, layer1_y-0.15, 'Feed Forward Network', 
        ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(9.75, layer1_y-0.35, 'FFN(x) = ReLU(xW₁+b₁)W₂+b₂', 
        ha='center', va='center', fontsize=7, family='monospace')

# Layer 2 (略小)
layer2_y = y_encoder-0.7
mha2 = FancyBboxPatch((8, layer2_y+0.15), 3.5, 0.4,
                      boxstyle="round,pad=0.05",
                      facecolor='#FFB347', edgecolor='black', linewidth=1.5)
ax.add_patch(mha2)
ax.text(9.75, layer2_y+0.35, 'Multi-Head Attention', 
        ha='center', va='center', fontsize=8, fontweight='bold')

ff2 = FancyBboxPatch((8, layer2_y-0.35), 3.5, 0.4,
                     boxstyle="round,pad=0.05",
                     facecolor='#77DD77', edgecolor='black', linewidth=1.5)
ax.add_patch(ff2)
ax.text(9.75, layer2_y-0.15, 'Feed Forward Network', 
        ha='center', va='center', fontsize=8, fontweight='bold')

# Add & Norm标注
for y in [layer1_y+0.05, layer1_y-0.55, layer2_y+0.0, layer2_y-0.5]:
    ax.text(11.7, y, 'Add &\nNorm', ha='center', va='center', 
           fontsize=6, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

# ==================== 4. Latent Space ====================
arrow3 = FancyArrowPatch((12, y_input), (13, y_input),
                        arrowstyle='->', mutation_scale=25, linewidth=3, color='red')
ax.add_patch(arrow3)

latent_box = FancyBboxPatch((13, y_encoder-0.8), 2, 2,
                           boxstyle="round,pad=0.15",
                           facecolor=color_latent, edgecolor='red', linewidth=3)
ax.add_patch(latent_box)

ax.text(14, y_encoder+0.9, 'Latent Space', ha='center', va='center', 
        fontsize=12, fontweight='bold', color='red')
ax.text(14, y_encoder+0.5, 'Dimension: 16', ha='center', va='center', fontsize=10)
ax.text(14, y_encoder+0.15, 'Dense(64→16)', ha='center', va='center', fontsize=8)
ax.text(14, y_encoder-0.2, 'z = Encoder(x)', ha='center', va='center', 
        fontsize=9, family='monospace', style='italic')
ax.text(14, y_encoder-0.5, '✨ Compressed\nRepresentation', 
        ha='center', va='center', fontsize=8, color='red', fontweight='bold')

# 绘制潜在向量示意
for i in range(16):
    circle = Circle((13.3 + (i%8)*0.2, y_encoder-0.95 - (i//8)*0.2), 0.08,
                   facecolor=plt.cm.plasma(i/16), edgecolor='black', linewidth=0.5)
    ax.add_patch(circle)

# ==================== 5. Transformer Decoder ====================
y_decoder = 8

arrow4 = FancyArrowPatch((14, y_encoder-1.2), (14, y_decoder+1.5),
                        arrowstyle='->', mutation_scale=25, linewidth=3, color='black')
ax.add_patch(arrow4)

decoder_box = FancyBboxPatch((7.5, y_decoder-1.2), 4.5, 2.8,
                            boxstyle="round,pad=0.2",
                            facecolor=color_decoder, edgecolor='black', linewidth=3)
ax.add_patch(decoder_box)

ax.text(9.75, y_decoder+1.4, 'Transformer Decoder', ha='center', va='center', 
        fontsize=13, fontweight='bold')
ax.text(9.75, y_decoder+1.05, '(2 Layers, Mirror Encoder)', ha='center', va='center', fontsize=9)

# Decoder layers (简化版)
for i, layer_y in enumerate([y_decoder+0.4, y_decoder-0.7]):
    mha = FancyBboxPatch((8, layer_y+0.2), 3.5, 0.5,
                         boxstyle="round,pad=0.05",
                         facecolor='#87CEEB', edgecolor='black', linewidth=1.5)
    ax.add_patch(mha)
    ax.text(9.75, layer_y+0.45, 'Multi-Head Attention' if i==0 else 'MHA', 
            ha='center', va='center', fontsize=9 if i==0 else 8, fontweight='bold')
    
    ff = FancyBboxPatch((8, layer_y-0.4), 3.5, 0.5,
                        boxstyle="round,pad=0.05",
                        facecolor='#98FB98', edgecolor='black', linewidth=1.5)
    ax.add_patch(ff)
    ax.text(9.75, layer_y-0.15, 'Feed Forward' if i==0 else 'FFN', 
            ha='center', va='center', fontsize=9 if i==0 else 8, fontweight='bold')

ax.text(9.75, y_decoder-1.4, 'Dense(16→64→100×6)', ha='center', va='center', fontsize=8)

# ==================== 6. Output ====================
arrow5 = FancyArrowPatch((9.75, y_decoder-1.5), (9.75, 5.5),
                        arrowstyle='->', mutation_scale=25, linewidth=3, color='black')
ax.add_patch(arrow5)

output_box = FancyBboxPatch((7.5, 4.5), 4.5, 1,
                           boxstyle="round,pad=0.1",
                           facecolor=color_output, edgecolor='black', linewidth=2)
ax.add_patch(output_box)

ax.text(9.75, 5.2, 'Reconstructed Output', ha='center', va='center', 
        fontsize=12, fontweight='bold')
ax.text(9.75, 4.9, 'Shape: (Batch, 100, 6)', ha='center', va='center', fontsize=9)
ax.text(9.75, 4.6, 'x̂ ≈ x', ha='center', va='center', 
        fontsize=10, family='monospace', style='italic')

# ==================== 7. Loss Function ====================
loss_box = FancyBboxPatch((12.5, 4.5), 3, 1,
                         boxstyle="round,pad=0.1",
                         facecolor='#FFDAB9', edgecolor='black', linewidth=2)
ax.add_patch(loss_box)

ax.text(14, 5.2, 'Loss Function', ha='center', va='center', 
        fontsize=11, fontweight='bold')
ax.text(14, 4.9, 'MSE(x, x̂)', ha='center', va='center', fontsize=9)
ax.text(14, 4.6, 'L = ||x - x̂||²', ha='center', va='center', 
        fontsize=9, family='monospace')

# 箭头连接
arrow_loss1 = FancyArrowPatch((9.75, 5.3), (12.5, 5.0),
                             arrowstyle='-', linestyle='--', linewidth=1.5, color='gray')
ax.add_patch(arrow_loss1)

arrow_loss2 = FancyArrowPatch((2, 11.2), (13, 5.3),
                             arrowstyle='-', linestyle='--', linewidth=1.5, color='gray')
ax.add_patch(arrow_loss2)

# ==================== 8. Clustering ====================
y_cluster = 2.5

arrow6 = FancyArrowPatch((14, y_encoder-1.2), (14, y_cluster+1.2),
                        arrowstyle='->', mutation_scale=30, linewidth=4, color='blue')
ax.add_patch(arrow6)

cluster_box = FancyBboxPatch((12, y_cluster-0.8), 4, 1.8,
                            boxstyle="round,pad=0.15",
                            facecolor=color_cluster, edgecolor='blue', linewidth=3)
ax.add_patch(cluster_box)

ax.text(14, y_cluster+0.8, 'K-Means Clustering', ha='center', va='center', 
        fontsize=13, fontweight='bold', color='blue')
ax.text(14, y_cluster+0.4, 'Input: Latent features (N × 16)', 
        ha='center', va='center', fontsize=9)
ax.text(14, y_cluster+0.05, 'K = 4 clusters', ha='center', va='center', fontsize=10)
ax.text(14, y_cluster-0.3, 'Output: Cluster labels', ha='center', va='center', fontsize=9)
ax.text(14, y_cluster-0.6, '[C0, C1, C2, C3]', ha='center', va='center', 
        fontsize=9, family='monospace', style='italic')

# ==================== 9. Final Results ====================
y_final = 0.3

arrow7 = FancyArrowPatch((14, y_cluster-0.9), (14, y_final+0.5),
                        arrowstyle='->', mutation_scale=25, linewidth=3, color='black')
ax.add_patch(arrow7)

final_box = FancyBboxPatch((10.5, y_final-0.5), 7, 1,
                          boxstyle="round,pad=0.1",
                          facecolor='#90EE90', edgecolor='black', linewidth=2)
ax.add_patch(final_box)

ax.text(14, y_final+0.25, '🏆 Driving Behavior Clusters', ha='center', va='center', 
        fontsize=12, fontweight='bold')
ax.text(14, y_final-0.15, 'C0: Highway Cruise | C1: Urban Driving | C2: Aggressive | C3: Congestion', 
        ha='center', va='center', fontsize=8)

# ==================== 左侧：关键参数总结 ====================
params_y = 8
params_box = FancyBboxPatch((0.3, params_y-3.5), 3.5, 4,
                           boxstyle="round,pad=0.2",
                           facecolor='#F0F8FF', edgecolor='black', linewidth=2)
ax.add_patch(params_box)

ax.text(2, params_y+0.3, 'Model Parameters', ha='center', va='center', 
        fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue'))

param_text = """
Architecture:
• Input: 100 × 6 sequences
• Positional Encoding: Sinusoidal
• Encoder: 2 Transformer layers
  - d_model = 64
  - nhead = 4 (d_k = 16)
  - d_ff = 256
  - dropout = 0.1
• Latent: 16 dimensions
• Decoder: Mirror encoder

Training:
• Optimizer: Adam (lr=0.001)
• Loss: MSE
• Epochs: 30
• Batch size: 32
• Early stopping: patience=15

Clustering:
• Algorithm: K-Means
• K = 4
• Init: k-means++
• n_init: 20
"""

ax.text(2, params_y-2, param_text, ha='left', va='top', 
        fontsize=7, family='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ==================== 右侧：数据流动标注 ====================
flow_y = 7
flow_box = FancyBboxPatch((16.5, flow_y-2.5), 3.2, 3.5,
                         boxstyle="round,pad=0.2",
                         facecolor='#FFF5EE', edgecolor='black', linewidth=2)
ax.add_patch(flow_box)

ax.text(18.1, flow_y+0.8, 'Data Flow', ha='center', va='center', 
        fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightyellow'))

flow_text = """
1️⃣ Raw Sequence
   (Batch, 100, 6)
   ↓
2️⃣ + Positional Encoding
   ↓
3️⃣ Transformer Encoding
   - Self-Attention
   - Feed Forward
   - Layer Norm
   ↓
4️⃣ Latent Representation
   (Batch, 16) ← KEY!
   ↓
5️⃣ Transformer Decoding
   ↓
6️⃣ Reconstruction
   (Batch, 100, 6)
   
   [Branch to Clustering]
   ↓
7️⃣ K-Means on Latent
   ↓
8️⃣ Cluster Labels
   [0, 1, 2, 3]
"""

ax.text(18.1, flow_y-1.8, flow_text, ha='left', va='top', 
        fontsize=7, family='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ==================== 图例 ====================
legend_elements = [
    mpatches.Patch(color=color_input, label='Input Layer', edgecolor='black'),
    mpatches.Patch(color=color_encoder, label='Encoder', edgecolor='black'),
    mpatches.Patch(color=color_latent, label='Latent Space', edgecolor='red'),
    mpatches.Patch(color=color_decoder, label='Decoder', edgecolor='black'),
    mpatches.Patch(color=color_cluster, label='Clustering', edgecolor='blue'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9, 
         bbox_to_anchor=(0.02, 0.98), framealpha=0.9)

plt.tight_layout()
plt.savefig('./results/transformer_ae_architecture.png', dpi=300, bbox_inches='tight')
print("✅ Transformer-AE架构图已保存: ./results/transformer_ae_architecture.png")
plt.close()

print("\n架构图生成完成！")
print("展示了从输入序列 → Transformer编码 → 潜在空间 → 解码 → 聚类的完整流程")