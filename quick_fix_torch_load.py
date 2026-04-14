"""快速修复torch.load"""
with open('dual_channel_balanced_clustering_fixed.py', 'r') as f:
    content = f.read()

# 替换所有torch.load
content = content.replace(
    "torch.load('./results/balanced_clustering_best.pth')",
    "torch.load('./results/balanced_clustering_best.pth', weights_only=False)"
)

# 保存
with open('dual_channel_balanced_clustering_fixed.py', 'w') as f:
    f.write(content)

print("✅ 修复完成！")