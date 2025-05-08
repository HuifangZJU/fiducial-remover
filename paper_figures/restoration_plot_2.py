# === 1. Import packages ===
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.colors import to_rgba

# === 2. Define colors ===
# lama_fill_color = '#1c8041'
# sd_fill_color = '#e55709'
# dip_fill_color = '#0072B2'
# base = ['#c6d182','#e0c7e3','#eae0e9','#ae98b6','#846e89']

lama_fill_color = '#c6d182'
sd_fill_color = '#ae98b6'
dip_fill_color = '#846e89'

fill_alpha = 1.0  # inside bar transparency
edge_alpha = 1.0  # solid border
thicker_edge_width = 1.5  # bold border width

# === 3. Helper function ===
def color_with_alpha(color, alpha_value):
    rgba = list(to_rgba(color))
    rgba[-1] = alpha_value
    return tuple(rgba)

# === 4. Define the data ===

# X-axis labels
common_labels = ['608x608', '1024x992', '1216x1184', '1408x1376']
x = np.arange(len(common_labels))

# Runtime data (seconds)
lama_runtime_common = [2.0858, 1.0272, 0.9963, 1.1067]
sd_runtime_common = [7.2401, 27.8179, 52.0599, 90.9860]
dip_runtime_common = [140.2910, 374.5532, 538.8126, 712.5731]

# GPU memory data (GB)
lama_gpu_mem_common = [922.63/1024, 1224.74/1024, 1650.66/1024, 2153.99/1024]
sd_gpu_mem_common = [3485.46/1024, 4943.86/1024, 5938.07/1024, 7131.28/1024]
dip_gpu_mem_common = [1080.45/1024, 2745.45/1024, 3838.45/1024, 5123.40/1024]

# Max pixel sizes
lama_max_pixel = 3328 * 3264
dip_max_pixel = 1664 * 1632
sd_max_pixel = 1408 * 1376

# === 5. Plot ===

fig = plt.figure(figsize=(22, 5.8))
plt.rcParams.update({'font.size': 20})
gs = gridspec.GridSpec(1, 3, width_ratios=[0.7, 1.2, 1.2])

ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])

# 5.1 Max Image Size plot
bars = ax0.bar(
    ['LaMa (Vispro)', 'DIP', 'Stable Diffusion'],
    [lama_max_pixel, dip_max_pixel, sd_max_pixel],
    width=0.5,
    color=[color_with_alpha(lama_fill_color, fill_alpha),
           color_with_alpha(dip_fill_color, fill_alpha),
           color_with_alpha(sd_fill_color, fill_alpha)],
    edgecolor=[color_with_alpha(lama_fill_color, edge_alpha),
               color_with_alpha(dip_fill_color, edge_alpha),
               color_with_alpha(sd_fill_color, edge_alpha)],
    linewidth=thicker_edge_width,
)
ax0.set_ylabel('Pixel count')
ax0.set_title('Max image size (8GB GPU)')
ax0.set_yticks([lama_max_pixel, dip_max_pixel, sd_max_pixel])
ax0.set_yticklabels(['3328x3264', '1664x1632', '1408x1376'], rotation=45, ha='right', fontsize=16)
ax0.tick_params(axis='x')
# Explicitly set font size for x-axis labels
ax0.tick_params(axis='x', labelsize=16)

# 5.2 GPU Memory plot
bars1 = ax1.bar(x - 0.25, lama_gpu_mem_common, width=0.25,
                label='LaMa', color=color_with_alpha(lama_fill_color, fill_alpha),
                edgecolor=color_with_alpha(lama_fill_color, edge_alpha), linewidth=thicker_edge_width)
bars2 = ax1.bar(x, dip_gpu_mem_common, width=0.25,
                label='DIP', color=color_with_alpha(dip_fill_color, fill_alpha),
                edgecolor=color_with_alpha(dip_fill_color, edge_alpha), linewidth=thicker_edge_width)
bars3 = ax1.bar(x + 0.25, sd_gpu_mem_common, width=0.25,
                label='Stable Diffusion', color=color_with_alpha(sd_fill_color, fill_alpha),
                edgecolor=color_with_alpha(sd_fill_color, edge_alpha), linewidth=thicker_edge_width)

ax1.set_ylabel('GPU memory (GB)')
ax1.set_title('Memory usage (GPU)')
ax1.set_xticks(x)
ax1.set_xticklabels(common_labels, ha='center', fontsize=16)
# ax1.legend()

# 5.3 Runtime plot
bars1 = ax2.bar(x - 0.25, lama_runtime_common, width=0.25,
                label='LaMa (Vispro)', color=color_with_alpha(lama_fill_color, fill_alpha),
                edgecolor=color_with_alpha(lama_fill_color, edge_alpha), linewidth=thicker_edge_width)
bars2 = ax2.bar(x, sd_runtime_common, width=0.25,
                label='Stable Diffusion', color=color_with_alpha(sd_fill_color, fill_alpha),
                edgecolor=color_with_alpha(sd_fill_color, edge_alpha), linewidth=thicker_edge_width)
bars3 = ax2.bar(x + 0.25, dip_runtime_common, width=0.25,
                label='DIP', color=color_with_alpha(dip_fill_color, fill_alpha),
                edgecolor=color_with_alpha(dip_fill_color, edge_alpha), linewidth=thicker_edge_width)

ax2.set_ylabel('Runtime (seconds)')
ax2.set_title('Runtime (Log scale)')
ax2.set_xticks(x)
ax2.set_xticklabels(common_labels,ha='center', fontsize=16)
ax2.set_yscale('log')
ax2.legend(frameon=True, fontsize=16, bbox_to_anchor=(1, 0.7))

bar_heights=[]
# Add value labels to runtime plot
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        bar_heights.append(height)
        if not np.isnan(height):
            ax2.annotate(f'{height:.1f}',
                         xy=(bar.get_x() + bar.get_width() / 2.5, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=14)

ax2.set_ylim(0, max(bar_heights) * 1.5)
# Final layout

ax0.grid(axis='y', linestyle='--', alpha=0.4)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax2.grid(axis='y', linestyle='--', alpha=0.4)


plt.tight_layout()
plt.savefig('./figures/11.png', dpi=300)
plt.show()
