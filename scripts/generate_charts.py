"""
Generate the 'Money Chart' for the paper
Compares PyTorch (baseline) vs INT8-416 (speed) vs INT8-640 (accuracy)
"""

import matplotlib.pyplot as plt
import numpy as np

# V3 Benchmark Data
models = ['PyTorch\n(Baseline)', 'INT8 @ 416\n(Speed)', 'INT8 @ 640\n(Accuracy)']

# mAP@0.5 (test set)
map_values = [96.86, 93.19, 97.22]

# FPS (720p pipeline)
fps_values = [4.1, 13.0, 7.4]

# Model Size (MB)
size_values = [5.2, 3.2, 3.2]

# Speedup vs PyTorch
speedup_values = [1.0, 3.2, 1.8]  # Based on FPS ratio

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Colors
colors = ['#4472C4', '#70AD47', '#ED7D31']

# Chart 1: FPS Comparison
ax1 = axes[0]
bars1 = ax1.bar(models, fps_values, color=colors, edgecolor='black', linewidth=1.2)
ax1.set_ylabel('FPS (720p Video)', fontsize=12, fontweight='bold')
ax1.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 16)
ax1.axhline(y=10, color='red', linestyle='--', linewidth=1.5, label='Real-time threshold (10 FPS)')
ax1.legend(loc='upper right', fontsize=9)

# Add value labels on bars
for bar, val in zip(bars1, fps_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
             f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Chart 2: Accuracy Comparison
ax2 = axes[1]
bars2 = ax2.bar(models, map_values, color=colors, edgecolor='black', linewidth=1.2)
ax2.set_ylabel('mAP@0.5 (%)', fontsize=12, fontweight='bold')
ax2.set_title('Detection Accuracy Comparison', fontsize=14, fontweight='bold')
ax2.set_ylim(90, 100)

# Add value labels on bars
for bar, val in zip(bars2, map_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
             f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Styling
for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)

plt.tight_layout()

# Save chart
output_path = 'reports/model_comparison_chart.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✅ Chart saved to: {output_path}")

plt.close()

# Also create a single combined chart for simpler use
fig2, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))
width = 0.35

# Create grouped bars
bars_fps = ax.bar(x - width/2, fps_values, width, label='FPS (720p)', color='#4472C4', edgecolor='black')
bars_map = ax.bar(x + width/2, [m - 90 for m in map_values], width, label='mAP@0.5 - 90%', color='#70AD47', edgecolor='black')

ax.set_ylabel('Value', fontsize=12, fontweight='bold')
ax.set_title('Speed vs Accuracy Trade-off', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='upper right')
ax.axhline(y=10, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
ax.text(2.5, 10.5, 'Real-time threshold', fontsize=9, color='red')

# Add annotations
for i, (fps, mAP) in enumerate(zip(fps_values, map_values)):
    ax.text(i - width/2, fps + 0.3, f'{fps:.1f}', ha='center', fontsize=10, fontweight='bold')
    ax.text(i + width/2, mAP - 90 + 0.3, f'{mAP:.1f}%', ha='center', fontsize=10, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('reports/speed_vs_accuracy_tradeoff.png', dpi=150, bbox_inches='tight', facecolor='white')
print("✅ Trade-off chart saved to: reports/speed_vs_accuracy_tradeoff.png")

print("\n📊 Charts generated successfully!")
