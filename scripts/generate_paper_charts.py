"""
PAPER CHART GENERATION
Run AFTER collect_paper_data.py

Generates all charts for the paper:
1. Pareto Scatter Plot (FPS vs mAP)
2. Speedup Bar Chart (6x claim)
3. Latency Breakdown Stacked Bar
4. Class Imbalance Histogram
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load collected data
print("Loading paper data...")
with open('reports/paper_data.json') as f:
    data = json.load(f)

# Create output directory
output_dir = Path('reports/paper_figures')
output_dir.mkdir(exist_ok=True)

# ============================================================
# CHART 1: PARETO SCATTER PLOT (The "Executive Decision" Chart)
# ============================================================
print("Generating Chart 1: Pareto Scatter Plot...")

fig, ax = plt.subplots(figsize=(10, 7))

pareto = data['pareto_data']

# Data points
points = [
    ('PyTorch_FP32', pareto['PyTorch_FP32']['fps'], pareto['PyTorch_FP32']['mAP'], 'gray', 's', 150),
    ('INT8_640', pareto['INT8_640']['fps'], pareto['INT8_640']['mAP'], 'orange', '^', 200),
    ('INT8_416', pareto['INT8_416']['fps'], pareto['INT8_416']['mAP'], 'green', 'o', 250),
]

for name, fps, mAP, color, marker, size in points:
    ax.scatter(fps, mAP, c=color, marker=marker, s=size, edgecolors='black', linewidth=1.5, zorder=5)

# Add labels
ax.annotate('PyTorch\n(Too Slow)', (4.1, 96.86), textcoords="offset points", 
            xytext=(-40, 20), fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='gray'))

ax.annotate('INT8 @ 640\n(Best Accuracy)', (7.4, 97.22), textcoords="offset points", 
            xytext=(50, 10), fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='orange'))

ax.annotate('INT8 @ 416\n★ SWEET SPOT', (13.0, 93.19), textcoords="offset points", 
            xytext=(30, -30), fontsize=11, ha='center', fontweight='bold', color='green',
            arrowprops=dict(arrowstyle='->', color='green', lw=2))

# Real-time threshold line
ax.axvline(x=10, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.text(10.2, 92.5, 'Real-time\nThreshold\n(10 FPS)', fontsize=9, color='red', va='top')

# Styling
ax.set_xlabel('Inference Speed (FPS)', fontsize=14, fontweight='bold')
ax.set_ylabel('Detection Accuracy (mAP@0.5 %)', fontsize=14, fontweight='bold')
ax.set_title('Accuracy vs. Speed Trade-off', fontsize=16, fontweight='bold')
ax.set_xlim(0, 16)
ax.set_ylim(92, 98)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
legend_elements = [
    plt.scatter([], [], c='gray', marker='s', s=100, label='PyTorch FP32 (Baseline)'),
    plt.scatter([], [], c='orange', marker='^', s=100, label='INT8 @ 640 (Accuracy)'),
    plt.scatter([], [], c='green', marker='o', s=100, label='INT8 @ 416 (Production)')
]
ax.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'fig1_pareto_scatter.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Saved: fig1_pareto_scatter.png")

# ============================================================
# CHART 2: SPEEDUP BAR CHART (The "6x Claim" Chart)
# ============================================================
print("Generating Chart 2: Speedup Bar Chart...")

fig, ax = plt.subplots(figsize=(10, 6))

speedup = data['speedup_data']
models = ['PyTorch\nFP32', 'ONNX\nRuntime', 'OpenVINO\nFP32', 'OpenVINO\nINT8']
fps_values = [speedup['PyTorch']['fps'], speedup['ONNX']['fps'], 
              speedup['OpenVINO_FP32']['fps'], speedup['OpenVINO_INT8']['fps']]
speedup_values = [speedup['PyTorch']['speedup'], speedup['ONNX']['speedup'], 
                  speedup['OpenVINO_FP32']['speedup'], speedup['OpenVINO_INT8']['speedup']]

# Colors: Red for baseline, gradient to green for best
colors = ['#D32F2F', '#FFA726', '#66BB6A', '#2E7D32']

bars = ax.bar(models, fps_values, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, fps, sp in zip(bars, fps_values, speedup_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, 
            f'{fps:.1f} FPS\n({sp:.1f}×)', 
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Inference Speed (FPS)', fontsize=14, fontweight='bold')
ax.set_title('Quantization Pipeline Speedup', fontsize=16, fontweight='bold')
ax.set_ylim(0, 32)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add 6x annotation
ax.annotate('6× Faster!', xy=(3, 25.1), xytext=(2.5, 28),
            fontsize=14, fontweight='bold', color='#2E7D32',
            arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=2))

plt.tight_layout()
plt.savefig(output_dir / 'fig2_speedup_bar.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Saved: fig2_speedup_bar.png")

# ============================================================
# CHART 3: LATENCY BREAKDOWN STACKED BAR
# ============================================================
print("Generating Chart 3: Latency Breakdown...")

fig, ax = plt.subplots(figsize=(10, 4))

latency = data['latency_breakdown']

# Data
categories = ['INT8 Pipeline']
preprocess = [latency['preprocess_ms']]
inference = [latency['inference_ms']]
postprocess = [latency['postprocess_ms']]
render = [latency['render_ms']]

# Stack
bar_width = 0.5
x = np.arange(len(categories))

p1 = ax.barh(x, preprocess, bar_width, label=f"Preprocess ({latency['preprocess_ms']:.1f}ms)", color='#42A5F5')
p2 = ax.barh(x, inference, bar_width, left=preprocess, label=f"Inference ({latency['inference_ms']:.1f}ms)", color='#66BB6A')
p3 = ax.barh(x, postprocess, bar_width, left=[p+i for p,i in zip(preprocess, inference)], 
             label=f"Postprocess ({latency['postprocess_ms']:.1f}ms)", color='#FFA726')
p4 = ax.barh(x, render, bar_width, left=[p+i+pp for p,i,pp in zip(preprocess, inference, postprocess)], 
             label=f"Render ({latency['render_ms']:.1f}ms)", color='#EF5350')

ax.set_xlabel('Time (ms)', fontsize=14, fontweight='bold')
ax.set_title('End-to-End Pipeline Latency Breakdown', fontsize=16, fontweight='bold')
ax.set_yticks(x)
ax.set_yticklabels(categories)
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(0, latency['total_ms'] + 10)

# Add total time annotation
total = latency['total_ms']
ax.annotate(f'Total: {total:.1f}ms\n({1000/total:.1f} FPS)', 
            xy=(total, 0), xytext=(total + 5, 0.2),
            fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black'))

plt.tight_layout()
plt.savefig(output_dir / 'fig3_latency_breakdown.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Saved: fig3_latency_breakdown.png")

# ============================================================
# CHART 4: CLASS IMBALANCE HISTOGRAM
# ============================================================
print("Generating Chart 4: Class Imbalance Histogram...")

fig, ax = plt.subplots(figsize=(12, 6))

top_5 = list(data['top_5_classes'].items())
bottom_5 = list(data['bottom_5_classes'].items())

# Combine for chart (Top 5 + gap + Bottom 5)
all_classes = top_5 + bottom_5
class_names = [c[0] for c in all_classes]
class_counts = [c[1] for c in all_classes]

# Shorten names for display
short_names = [n.replace('speedLimit', 'SL').replace('rampSpeedAdvisory', 'RSA')
               .replace('pedestrianCrossing', 'PedXing').replace('signalAhead', 'Signal') 
               for n in class_names]

colors = ['#4CAF50']*5 + ['#F44336']*5  # Green for top, red for bottom

bars = ax.bar(short_names, class_counts, color=colors, edgecolor='black')

# Add value labels
for bar, count in zip(bars, class_counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 20, 
            str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel('Class Name', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Training Instances', fontsize=14, fontweight='bold')
ax.set_title('Class Distribution: Top 5 vs Bottom 5', fontsize=16, fontweight='bold')
ax.set_xticklabels(short_names, rotation=45, ha='right')

# Add imbalance ratio
top_count = sum([c[1] for c in top_5])
bottom_count = sum([c[1] for c in bottom_5])
ratio = top_count / max(bottom_count, 1)
ax.text(0.95, 0.95, f'Imbalance Ratio: {ratio:.0f}:1', 
        transform=ax.transAxes, fontsize=12, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(output_dir / 'fig4_class_imbalance.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  ✅ Saved: fig4_class_imbalance.png")

# ============================================================
# DONE
# ============================================================
print("\n" + "="*60)
print("ALL CHARTS GENERATED!")
print("="*60)
print(f"Output directory: {output_dir.absolute()}")
print("\nFiles:")
for f in output_dir.glob('*.png'):
    print(f"  - {f.name}")
