import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.ticker import MultipleLocator, FuncFormatter
# Define legend names and color palette
legend_names = [
    'bUnwarpJ (Original image)',
    'bUnwarpJ (Vispro)',
    'SimpleITK (Original image)',
    'SimpleITK (Vispro)',
    # 'Initial position (vispro)',
    # 'Initial position (original)'
]

# Base color palette
base = ['#c6d182', '#e0c7e3', '#eae0e9', '#ae98b6', '#846e89']
sorted_colors = [
    mcolors.to_hex(np.array(mcolors.to_rgb(base[4])) * 0.8),  # Registered BJ (original)
    base[0],  # Registered BJ (vispro) - green
    base[3],  # Registered ITK (original)
    mcolors.to_hex(np.array(mcolors.to_rgb(base[0])) * 0.85),  # Registered ITK (vispro)
    # base[4],  # Initial position (vispro)
    # base[2]  # Initial position (original)
]

# Visium and Cytassist data: SSIM (row 0), MI (row 1)
# visium_data = np.array([
#     [0.6732, 0.6749, 0.5936, 0.5809, 0.6289, 0.5272],  # SSIM
#     [0.5321, 0.5077, 0.4599, 0.4290, 0.3323, 0.2890]   # MI
# ])
# cytassist_data = np.array([
#     [0.8164, 0.7969, 0.7905, 0.7776, 0.5700, 0.5279],  # SSIM
#     [0.7134, 0.6777, 0.6538, 0.6385, 0.3187, 0.2873]   # MI
# ])

visium_data = np.array([
    [0.5936,0.6732, 0.5809,  0.6749],  # SSIM
    [0.4599,0.5321,  0.4290, 0.5077]   # MI
])
# do not include the all black area
# cytassist_data = np.array([
#     [0.8164,  0.7905,0.7969, 0.7776],  # SSIM
#     [0.7134,  0.6538, 0.6777,0.6385]   # MI
# ])
cytassist_data = np.array([
    [0.6340,0.6588, 0.6236,  0.6488],  # SSIM
    [0.6129, 0.6907,0.6180,  0.6695]   # MI
])

# Plotting setup
plt.rcParams.update({'font.size': 14})
fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
datasets = ['Visium', 'CytAssist']
data_groups = [visium_data, cytassist_data]
metrics = ['SSIM', 'MI']
x = np.arange(len(metrics))
# bar_width = 0.12
# offsets = np.linspace(-0.3, 0.3, len(legend_names))
bar_width = 0.13
offsets = np.linspace(-0.2, 0.2, len(legend_names))



# Plot bars for each dataset
for ax, data, title in zip(axs, data_groups, datasets):
    for i in range(len(legend_names)):
        ax.bar(
            x[0] + offsets[i], data[0, i],
            width=bar_width, color=sorted_colors[i],
            label=legend_names[i] if title == 'Visium' else "")
        ax.bar(
            x[1] + offsets[i], data[1, i],
            width=bar_width, color=sorted_colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title(title)
    ax.set_ylabel('Metric Value')
    ax.set_ylim(0.4, max(data.flatten()) + 0.03)
    # ax.set_ylim(0.4, 0.7)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))
    ax.grid(axis='y', linestyle='--', alpha=0.4)

# Add legend to the far right of axs[1]
handles, labels = axs[0].get_legend_handles_labels()
axs[1].legend(handles, labels, fontsize=12,loc='center left', bbox_to_anchor=(1.03, 0.5))

plt.tight_layout()
plt.savefig('./figures/12.png', dpi=300)
plt.show()
