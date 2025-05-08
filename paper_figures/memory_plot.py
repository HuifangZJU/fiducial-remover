import matplotlib.pyplot as plt

def plot_gpu():
    # Image sizes in 1000-pixel units
    x_vals = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # GPU memory usage (MB)
    gpu_marker = [189.77, 1611.73, 6412.70, 6412.70, 6406.50, 6406.50, 6406.50, 6406.50, 6406.50, 6406.50, 6406.50,
                  6406.50, 6406.50, 6406.50]
    gpu_inpaint = [597.04, 1277.55, 4754.98, 2435.98, 2512.01, 2588.30, 2798.11, 2989.41, 2969.79, 2951.67, 3027.45,
                   3065.93, 3447.22, 3580.12]
    gpu_bg = [396.29, 406.41, 1198.47, 4754.98, 4362.27, 4362.27, 4362.27, 4362.27, 4362.27, 4362.27, 4362.27, 4362.27,
              4362.27, 4362.27]
    gpu_seg = [11.12, 20.12, 812.76, 812.76, 54.64, 54.64, 54.64, 54.64, 54.64, 54.64, 54.64, 54.64, 54.64, 54.64]

    gpu_marker = [v / 1024 for v in gpu_marker]
    gpu_inpaint = [v / 1024 for v in gpu_inpaint]
    gpu_bg = [v / 1024 for v in gpu_bg]
    gpu_seg = [v / 1024 for v in gpu_seg]

    # Base colors
    colors = ['#501d8a', '#1c8041', '#e55709', '#1e6cb3']

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.rcParams.update({'font.size': 14})
    x_ticks = [1, 3, 4, 5,
               6, 7, 8, 9, 10, 11, 12]
    x_tick_labels = [
        '1',
        '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
    ]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=11)

    # Step 3: Manually overlay bold labels on 0.5, 2, and 13
    bold_ticks = {
        0.45: '0.5\nLow-res',
        2: '2\nHigh-res',
        13: '13\nMicroscope H&E'
    }

    for xpos, label in bold_ticks.items():
        ax.text(
            xpos,  # x position
            -0.034,  # y position in axes fraction (just below the tick)
            label,
            ha='center', va='top',
            fontsize=11, fontweight='bold',
            transform=ax.get_xaxis_transform()
        )


    ax.axvline(x=2, linestyle='--', color='gray', linewidth=1)

    ax.annotate(
        "Flat GPU usage\n(resize + patch)",

        xy=(2, 6),
        xytext=(2.8, 5),
        textcoords='data',
        arrowprops=dict(arrowstyle="->", color='black'),
        ha='left',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5)
    )

    # Plot CPU memory curves
    ax.plot(x_vals, gpu_marker, label='Fiducial marker detection', color=colors[0], marker='o')
    ax.plot(x_vals, gpu_inpaint, label='Image restoration', color=colors[1], marker='s')
    ax.plot(x_vals, gpu_bg, label='Tissue detection', color=colors[2], marker='^')
    ax.plot(x_vals, gpu_seg, label='Tissue segregation', color=colors[3], marker='d')

    # Axis and labels
    ax.set_title('Memory usage (GPU)')
    ax.set_xlabel('Image size (in 1000-pixel units)',fontsize=12)
    ax.set_ylabel('Peak memory (GB)',fontsize=12)
    ax.grid(True)
    # ax.legend()

    plt.tight_layout()
    plt.savefig('./figures/8.png', dpi=300)
    plt.show()



def plot_cpu():
    # Image sizes in 1000-pixel units
    x_vals = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

    # CPU memory usage (MB) from provided data
    cpu_marker = [3.00, 9.77, 35.79, 99.84, 153.04, 221.69, 305.60, 404.76, 519.19, 648.88, 793.82, 954.02, 1129.47,
                  1320.20]
    cpu_inpaint = [13.27, 40.73, 185.76, 457.72, 709.99, 984.01, 1592.15, 2176.76, 2289.30, 2422.81, 2801.56, 3111.38,
                   4249.45, 4825.14]
    cpu_bg = [21.22, 77.46, 293.36, 397.67, 304.92, 305.01, 344.63, 444.13, 558.22, 687.90, 832.84, 993.05, 1168.50,
              1359.23]
    cpu_seg = [95.26, 381.01, 1476.80, 1476.79, 1476.75, 1476.75, 1476.75, 1476.75, 1476.75, 1476.75, 1476.75, 1476.75,
               1476.75, 1476.75]

    # Convert MB to GB
    cpu_marker = [v / 1024 for v in cpu_marker]
    cpu_inpaint = [v / 1024 for v in cpu_inpaint]
    cpu_bg = [v / 1024 for v in cpu_bg]
    cpu_seg = [v / 1024 for v in cpu_seg]

    # Base colors
    colors = ['#501d8a', '#1c8041', '#e55709', '#1e6cb3']

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.rcParams.update({'font.size': 14})
    x_ticks = [1, 3, 4, 5,
               6, 7, 8, 9, 10, 11, 12]
    x_tick_labels = [
        '1',
        '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
    ]


    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=12)

    # Step 3: Manually overlay bold labels on 0.5, 2, and 13
    bold_ticks = {
        0.45: '0.5\nLow-res',
        2: '2\nHigh-res',
        13: '13\nMicroscope H&E'
    }

    for xpos, label in bold_ticks.items():
        ax.text(
            xpos,  # x position
            -0.037,  # y position in axes fraction (just below the tick)
            label,
            ha='center', va='top',
            fontsize=11, fontweight='bold',
            transform=ax.get_xaxis_transform()
        )

    # Plot CPU memory curves
    ax.plot(x_vals, cpu_marker, label='Fiducial marker detection', color=colors[0], marker='o')
    ax.plot(x_vals, cpu_inpaint, label='Image restoration', color=colors[1], marker='s')
    ax.plot(x_vals, cpu_bg, label='Tissue detection', color=colors[2], marker='^')
    ax.plot(x_vals, cpu_seg, label='Tissue segregation', color=colors[3], marker='d')

    # Axis and labels
    ax.set_title('Memory usage (CPU)')
    ax.set_xlabel('Image size (in 1000-pixel units)',fontsize=12)
    ax.set_ylabel('Peak memory (GB)',fontsize=12)
    ax.grid(True)
    ax.legend()


    plt.tight_layout()
    plt.savefig('./figures/10.png', dpi=300)
    plt.show()





plot_cpu()
# plot_gpu()