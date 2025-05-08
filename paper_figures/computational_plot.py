import matplotlib.pyplot as plt

# X-axis values (image size in 1000-pixel units)
x_vals = [0.5, 1, 2, 3, 4, 5,
          6, 7, 8, 9, 10, 11, 12, 13]

# GPU timing data
gpu_marker_detection = [0.0744, 0.0855, 0.3466, 0.4405, 0.3656, 0.4478,
                        0.5496, 0.6761, 0.8768, 1.0353, 1.2731, 1.4523,
                        1.7775, 1.9232]

gpu_image_restoration = [0.5329, 0.8273, 2.2235, 2.7848, 3.7816, 4.8874,
                         7.9500, 12.0507, 13.1479, 11.7828, 14.9136, 19.8283,
                         21.4358, 25.1185]

gpu_background_removal = [0.3337, 0.4120, 0.9548, 3.3629, 2.4927, 2.6592,
                          2.6911, 2.9110, 3.3127, 3.2395, 3.5277, 3.7970,
                          4.0150, 4.1573]

gpu_tissue_segregation = [0.0813, 0.3744, 1.5599, 1.6945, 1.7242, 1.8422,
                          1.9132, 2.0514, 2.4747, 2.9118, 2.8387, 2.7514,
                          2.9814, 3.1597]

# CPU timing data
cpu_marker_detection = [0.1336, 0.2744, 1.9668, 2.6137, 3.1113, 3.0259,
                        3.1240, 4.0976, 3.3261, 3.5147, 3.7321, 3.9490,
                        4.1211, 4.2764]

cpu_image_restoration = [2.0359, 6.8994, 20.5331, 55.5504, 76.4123, 95.0271,
                         165.7779, 218.1150, 213.3360, 208.2696, 252.2898,
                         248.6065, 352.3830, 428.7783]

cpu_background_removal = [0.6482, 0.8115, 1.1723, 31.1075, 29.1721, 29.2786,
                          28.6884, 34.3628, 28.6633, 29.0203, 32.6942,
                          27.5933, 29.7859, 29.5455]

cpu_tissue_segregation = [0.1156, 0.5394, 1.7746, 1.9868, 1.9102, 1.8689,
                          1.9288, 1.9425, 2.1238, 2.2462, 2.3404,
                          2.7353, 2.6893, 2.9399]

# Base colors
base_colors =['#501d8a','#1c8041','#e55709', "#1e6cb3"]
# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(16, 4))
plt.rcParams.update({'font.size': 14})
# Bold label formatting using LaTeX
# Step 1: Regular (non-bold) x-tick labels
x_ticks = [1, 3, 4, 5,
          6, 7, 8, 9, 10, 11, 12]
x_tick_labels = [
    '1',
     '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
]

# Step 2: Set ticks and labels
for ax in axs:
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels,fontsize=11)

# Step 3: Manually overlay bold labels on 0.5, 2, and 13
bold_ticks = {
    0.45: '0.5\nLow-res',
    2: '2\nHigh-res',
    13: '13\nMicroscope H&E'
}

for ax in axs:
    for xpos, label in bold_ticks.items():
        ax.text(
            xpos,                # x position
            -0.034,               # y position in axes fraction (just below the tick)
            label,
            ha='center', va='top',
            fontsize=11, fontweight='bold',
            transform=ax.get_xaxis_transform()
        )
# GPU subplot
axs[0].plot(x_vals, gpu_marker_detection, marker='o', label='Fiducial marker detection', color=base_colors[0])
axs[0].plot(x_vals, gpu_image_restoration, marker='s', label='Image restoration', color=base_colors[1])
axs[0].plot(x_vals, gpu_background_removal, marker='^', label='Tissue detection', color=base_colors[2])
axs[0].plot(x_vals, gpu_tissue_segregation, marker='d', label='Tissue segregation', color=base_colors[3])
axs[0].set_title('Runtime (GPU)')
axs[0].set_xlabel('Image size (in 1000-pixel units)')
# axs[0].set_xticks(x_ticks)  # Ensure all ticks from 0.5 to 13 are shown
# axs[0].set_xticklabels(x_tick_labels)
axs[0].set_ylabel('Time (seconds)',fontsize=12)
axs[0].grid(True)
# axs[0].legend(bbox_to_anchor=(1, 0.7),fontsize=12)
axs[0].legend()

# CPU subplot
axs[1].plot(x_vals, cpu_marker_detection, marker='o', label='Fiducial marker detection', color=base_colors[0])
axs[1].plot(x_vals, cpu_image_restoration, marker='s', label='Image restoration', color=base_colors[1])
axs[1].plot(x_vals, cpu_background_removal, marker='^', label='Tissue detection', color=base_colors[2])
axs[1].plot(x_vals, cpu_tissue_segregation, marker='d', label='Tissue segregation', color=base_colors[3])
axs[1].set_title('Runtime (CPU)')
axs[1].set_xlabel('Image size (in 1000-pixel units)',fontsize=12)
axs[1].set_ylabel('Time (seconds)',fontsize=12)
# axs[1].set_xticks(x_ticks)  # Ensure all ticks from 0.5 to 13 are shown
# axs[1].set_xticklabels(x_tick_labels)
axs[1].grid(True)
# axs[1].legend(bbox_to_anchor=(1, 0.7),fontsize=12)
axs[1].legend()

plt.tight_layout()
plt.savefig('./figures/9.png', dpi=300)
plt.show()
