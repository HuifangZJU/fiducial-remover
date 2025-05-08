import matplotlib.pyplot as plt

# Updated image resolutions
x_labels = [
    '608x608', '1024x992', '1216x1184', '1408x1376', '1664x1632',
    '2016x1984', '2400x2368', '3328x3264'
]
x_indices = list(range(len(x_labels)))

# base = ['#c6d182','#e0c7e3','#eae0e9','#ae98b6','#846e89']
# Colors
lama_color = '#1c8041'
sd_color = '#e55709'
dip_color = '#0072B2'
highlight_color = 'red'


# LaMa updated
lama_runtime = [2.0858, 1.0272, 0.9963, 1.1067, 1.3202, 1.6660, 2.3454,  4.1868]
lama_cpu_mem = [59.04, 39.59, 53.68, 70.77, 97.48, 141.57, 199.81,  377.19]
lama_gpu_mem = [922.63, 1224.74, 1650.66, 2153.99, 2938.43, 4231.01, 5928.04,  7163.88]
lama_cpu_mem = [v / 1024 for v in lama_cpu_mem]
lama_gpu_mem = [v / 1024 for v in lama_gpu_mem]

# Stable Diffusion
sd_runtime = [7.2401, 27.8179, 52.0599, 90.9860]
sd_cpu_mem = [9.66, 26.30, 37.69, 51.45]
sd_gpu_mem = [3485.46, 4943.86, 5938.07, 7131.28]
sd_cpu_mem = [v / 1024 for v in sd_cpu_mem]
sd_gpu_mem = [v / 1024 for v in sd_gpu_mem]

# DIP
dip_runtime = [140.2910, 374.5532, 538.8126,712.5731,972.2801]
dip_cpu_mem = [14.49, 38.39, 54.15,72.65,101.60]
dip_gpu_mem = [1080.45, 2745.45,3838.45, 5123.40,6797.57]
dip_cpu_mem = [v / 1024 for v in dip_cpu_mem]
dip_gpu_mem = [v / 1024 for v in dip_gpu_mem]
dip_indices = x_indices[:5]

# Plotting
plt.rcParams.update({'font.size': 14})
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
plot_order = [0, 1, 2]
titles = ['Inpainting Runtime', 'GPU Memory Usage', 'CPU Memory Usage']
y_labels = ['Time (s)', 'Peak Memory (GB)', 'Peak Memory (GB)']

for idx, ax_idx in enumerate(plot_order):
    ax = axs[idx]
    ax.set_title(titles[idx])
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, rotation=45, ha='right',fontsize=12)
    ax.grid(True)

    if ax_idx == 0:
        ax.plot(x_indices, lama_runtime, color=lama_color, marker='o', label='LaMa')
        ax.plot(x_indices[:4], sd_runtime, color=sd_color, marker='^', label='Stable Diffusion')
        ax.plot(dip_indices, dip_runtime, color=dip_color, marker='v', label='DIP')
        ax.scatter(x_indices[-1], lama_runtime[-1], color=highlight_color, marker='s', s=70, alpha=0.6, label='Max image size')
        ax.scatter(x_indices[3], sd_runtime[3], color=highlight_color, marker='s', s=70, alpha=0.6)
        ax.scatter(dip_indices[-1], dip_runtime[-1], color=highlight_color, marker='s', s=70, alpha=0.6)
        ax.set_ylabel(y_labels[idx])

    elif ax_idx == 2:
        ax.plot(x_indices, lama_cpu_mem, color=lama_color, marker='o', label='LaMa')
        ax.plot(x_indices[:4], sd_cpu_mem, color=sd_color, marker='^', label='Stable Diffusion')
        ax.plot(dip_indices, dip_cpu_mem, color=dip_color, marker='v', label='DIP')
        ax.scatter(x_indices[-1], lama_cpu_mem[-1], color=highlight_color, marker='s', s=70, alpha=0.6, label='Max image size')
        ax.scatter(x_indices[3], sd_cpu_mem[3], color=highlight_color, marker='s', s=70, alpha=0.6)
        ax.scatter(dip_indices[-1], dip_cpu_mem[-1], color=highlight_color, marker='s', s=70, alpha=0.6)
        ax.set_ylabel(y_labels[idx])

    elif ax_idx == 1:
        ax.plot(x_indices, lama_gpu_mem, color=lama_color, marker='o', label='LaMa')
        ax.plot(x_indices[:4], sd_gpu_mem, color=sd_color, marker='^', label='Stable Diffusion')
        ax.plot(dip_indices, dip_gpu_mem, color=dip_color, marker='v', label='DIP')
        ax.scatter(x_indices[-1], lama_gpu_mem[-1], color=highlight_color, marker='s', s=70, alpha=0.6, label='Max image size')
        ax.scatter(x_indices[3], sd_gpu_mem[3], color=highlight_color, marker='s', s=70, alpha=0.6)
        ax.scatter(dip_indices[-1], dip_gpu_mem[-1], color=highlight_color, marker='s', s=70, alpha=0.6)
        ax.set_ylabel(y_labels[idx])

    ax.legend(fontsize=12)

plt.tight_layout()
# plt.savefig('./figures/11.png', dpi=300)
plt.show()
