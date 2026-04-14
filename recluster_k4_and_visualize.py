"""
Fix Energy and Driving Behavior Visualizations - Full English Version
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os

# English only settings
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("Generating English visualizations...")

# Load data
save_dir = './results/dual_channel_cross_attn_k4'
driving_data = pd.read_csv(f'{save_dir}/clustered_results.csv')
features_df = pd.read_csv('./results/dual_channel_cross_attn_k4/clustered_results.csv')

data = features_df.merge(driving_data, on=['event_id', 'vehicle_id'])
driving_data = data[
    (data['speed_mean'] > 5) &
    (data['distance_total'] > 0.5) &
    (data['moving_ratio'] > 0.3)
].copy()

colors = sns.color_palette('Set2', 4)
viz_dir = f'{save_dir}/visualizations'
os.makedirs(viz_dir, exist_ok=True)

# Cluster names
cluster_names = {
    0: "Highway Cruise",
    1: "Urban Driving", 
    2: "Aggressive Driving",
    3: "Urban Congestion"
}

# ========== Energy Analysis ==========
print("\n🔋 Energy Analysis...")

fig = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. Average Power
ax1 = fig.add_subplot(gs[0, 0])
power_mean = driving_data.groupby('cluster')['power_mean'].mean().sort_index()
bars = ax1.bar(power_mean.index, power_mean.values,
              color=sns.color_palette('YlOrRd', 4))
ax1.set_xlabel('Cluster ID', fontsize=11)
ax1.set_ylabel('Average Power (kW)', fontsize=11)
ax1.set_title('Average Power by Cluster', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, power_mean.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

# 2. SOC Drop
ax2 = fig.add_subplot(gs[0, 1])
driving_data['soc_drop_abs'] = abs(driving_data['soc_drop_total'])
soc_drop = driving_data.groupby('cluster')['soc_drop_abs'].mean().sort_index()
bars = ax2.bar(soc_drop.index, soc_drop.values,
              color=sns.color_palette('Greens_r', 4))
ax2.set_xlabel('Cluster ID', fontsize=11)
ax2.set_ylabel('SOC Drop (%)', fontsize=11)
ax2.set_title('Energy Consumption (SOC)', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, soc_drop.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

# 3. Power Box Plot
ax3 = fig.add_subplot(gs[0, 2])
bp = ax3.boxplot([driving_data[driving_data['cluster']==c]['power_mean'].values 
                 for c in sorted(driving_data['cluster'].unique())],
                labels=[f'C{c}' for c in sorted(driving_data['cluster'].unique())],
                patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax3.set_ylabel('Power (kW)', fontsize=11)
ax3.set_title('Power Distribution', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 4. Power vs Speed
ax4 = fig.add_subplot(gs[1, 0])
for c in sorted(driving_data['cluster'].unique()):
    cluster_data = driving_data[driving_data['cluster'] == c]
    ax4.scatter(cluster_data['speed_mean'], cluster_data['power_mean'],
               alpha=0.6, s=30, label=f'C{c}', color=colors[c])
ax4.set_xlabel('Average Speed (km/h)', fontsize=11)
ax4.set_ylabel('Average Power (kW)', fontsize=11)
ax4.set_title('Power vs Speed', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# 5. Duration
ax5 = fig.add_subplot(gs[1, 1])
duration = driving_data.groupby('cluster')['duration_minutes'].mean().sort_index()
bars = ax5.bar(duration.index, duration.values,
              color=sns.color_palette('Purples', 4))
ax5.set_xlabel('Cluster ID', fontsize=11)
ax5.set_ylabel('Average Duration (min)', fontsize=11)
ax5.set_title('Usage Duration', fontsize=12, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, duration.values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

# 6. Energy vs Distance
ax6 = fig.add_subplot(gs[1, 2])
for c in sorted(driving_data['cluster'].unique()):
    cluster_data = driving_data[driving_data['cluster'] == c]
    ax6.scatter(cluster_data['distance_total'], cluster_data['soc_drop_abs'],
               alpha=0.6, s=30, label=f'C{c}', color=colors[c])
ax6.set_xlabel('Distance (km)', fontsize=11)
ax6.set_ylabel('SOC Drop (%)', fontsize=11)
ax6.set_title('Energy vs Distance', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3)

plt.suptitle('Transformer-AE (K=4) - Energy Analysis', fontsize=16, fontweight='bold')
plt.savefig(f'{viz_dir}/03_energy_analysis_en.png', dpi=300, bbox_inches='tight')
print("  ✅ 03_energy_analysis_en.png")
plt.close()

# ========== Driving Behavior ==========
print("\n⚡ Driving Behavior Analysis...")

fig = plt.figure(figsize=(20, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. Harsh Accel/Decel
ax1 = fig.add_subplot(gs[0, 0])
harsh_stats = driving_data.groupby('cluster')[['harsh_accel', 'harsh_decel']].mean()

x = np.arange(len(harsh_stats))
width = 0.35

ax1.bar(x - width/2, harsh_stats['harsh_accel'], width,
       label='Harsh Accel', color='#FF6B6B')
ax1.bar(x + width/2, harsh_stats['harsh_decel'], width,
       label='Harsh Decel', color='#4ECDC4')

ax1.set_xlabel('Cluster ID', fontsize=11)
ax1.set_ylabel('Count', fontsize=11)
ax1.set_title('Harsh Acceleration vs Deceleration', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'C{c}' for c in harsh_stats.index])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Acceleration Std
ax2 = fig.add_subplot(gs[0, 1])
acc_std = driving_data.groupby('cluster')['acc_std'].mean().sort_index()
bars = ax2.bar(acc_std.index, acc_std.values,
              color=sns.color_palette('Reds', 4))
ax2.set_xlabel('Cluster ID', fontsize=11)
ax2.set_ylabel('Acc Std Dev (m/s²)', fontsize=11)
ax2.set_title('Driving Aggressiveness', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, acc_std.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

# 3. Harsh Accel vs Decel Scatter
ax3 = fig.add_subplot(gs[0, 2])
for c in sorted(driving_data['cluster'].unique()):
    cluster_data = driving_data[driving_data['cluster'] == c]
    ax3.scatter(cluster_data['harsh_accel'], cluster_data['harsh_decel'],
               alpha=0.6, s=50, label=f'C{c}', color=colors[c])
ax3.set_xlabel('Harsh Accel Count', fontsize=11)
ax3.set_ylabel('Harsh Decel Count', fontsize=11)
ax3.set_title('Aggressive Driving Pattern', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Distance Distribution
ax4 = fig.add_subplot(gs[1, 0])
bp = ax4.boxplot([driving_data[driving_data['cluster']==c]['distance_total'].values 
                 for c in sorted(driving_data['cluster'].unique())],
                labels=[f'C{c}' for c in sorted(driving_data['cluster'].unique())],
                patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax4.set_ylabel('Distance (km)', fontsize=11)
ax4.set_title('Trip Distance Distribution', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

# 5. Moving Ratio
ax5 = fig.add_subplot(gs[1, 1])
moving_ratio = driving_data.groupby('cluster')['moving_ratio'].mean().sort_index() * 100
bars = ax5.bar(moving_ratio.index, moving_ratio.values,
              color=sns.color_palette('Greens', 4))
ax5.set_xlabel('Cluster ID', fontsize=11)
ax5.set_ylabel('Moving Ratio (%)', fontsize=11)
ax5.set_title('Moving Time Percentage', fontsize=12, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, moving_ratio.values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

# 6. Overall Aggressiveness
ax6 = fig.add_subplot(gs[1, 2])
driving_data['aggressiveness'] = (
    driving_data['harsh_accel'] + 
    driving_data['harsh_decel'] + 
    driving_data['acc_std'] * 10
)
aggr = driving_data.groupby('cluster')['aggressiveness'].mean().sort_index()
bars = ax6.bar(aggr.index, aggr.values,
              color=sns.color_palette('rocket', 4))
ax6.set_xlabel('Cluster ID', fontsize=11)
ax6.set_ylabel('Aggressiveness Score', fontsize=11)
ax6.set_title('Overall Driving Aggressiveness', fontsize=12, fontweight='bold')
ax6.grid(axis='y', alpha=0.3)

plt.suptitle('Transformer-AE (K=4) - Driving Behavior Analysis', fontsize=16, fontweight='bold')
plt.savefig(f'{viz_dir}/04_driving_behavior_en.png', dpi=300, bbox_inches='tight')
print("  ✅ 04_driving_behavior_en.png")
plt.close()

print("\n✅ English versions generated!")
print(f"📁 Saved in: {viz_dir}/")
print("   - 03_energy_analysis_en.png")
print("   - 04_driving_behavior_en.png")