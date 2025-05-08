import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Updated raw data:
# -------------------------------
sample_ids = [1, 17, 64, 66, 67, 68, 69, 70, 71, 73, 74, 75, 77, 78, 79, 82, 86, 88, 90, 100, 101, 109, 141, 149, 161, 163, 110]
iou_original = [0.7970420870330154, 0.47750940438871475, 0.7005235277830402, 0.9377413527979948,
                0.7269524565544774, 0.9872059197141493, 0.9903942026496448, 0.987395520278728,
                0.9803894863189868, 0.8709154631673311, 0.7765999276368255, 0.9131374033360825,
                0.9744698053135435, 0.2628399986439605, 0.9560934496561837, 0.9495888301554746,
                0.948494870870093, 0.9743101802270077, 0.9509542367119415, 0.7226799949274791,
                0.5343970454986732, 0.8563972419601955, 0.2929953523519311, 0.9095200671632383,
                0.8098705082550988, 0.6853200379641547,0.6109599447341992]
iou_vispro =   [0.8228085220257753, 0.92152646527865,    0.9573547242374869, 0.9794979380685895,
                0.9498663506392698, 0.9735302174857181, 0.9932179644277883, 0.9882103215319928,
                0.9881291654293101, 0.8578913946971166, 0.9914629520962752, 0.911556853079936,
                0.9796901129902424, 0.8579215314613488, 0.975951351265432, 0.980756609687277,
                0.9655627309497168, 0.9757670987835343, 0.9762188755271266, 0.9428469265405193,
                0.061842532114801764, 0.9004972690389743, 0.8639611195249464, 0.9824268234558586,
                0.9281309709918361, 0.9419629173470354,0.9669186646194712]

# -------------------------------
# Convert lists to numpy arrays:
# -------------------------------
sample_ids = np.array(sample_ids)
iou_original = np.array(iou_original)
iou_vispro = np.array(iou_vispro)

# Compute the IoU difference (Vispro - Original)
diff = iou_vispro - iou_original

# -------------------------------
# Filter: Keep only samples where Vispro outperforms Original
# -------------------------------
mask = iou_vispro > iou_original
sel_sample_ids = sample_ids[mask]
sel_iou_original = iou_original[mask]
sel_iou_vispro = iou_vispro[mask]
sel_diff = diff[mask]

# Optional: Reorder the selected samples by descending difference (for clear visualization)
order = np.argsort(sel_diff)[::-1]
sel_sample_ids = sel_sample_ids[order]
sel_iou_original = sel_iou_original[order]
sel_iou_vispro = sel_iou_vispro[order]
sel_diff = sel_diff[order]

# -------------------------------
# Plotting
# -------------------------------
num_selected = len(sel_sample_ids)
ind = np.arange(num_selected)
width = 0.35

# Choose two consistent base colors:
color_vispro = '#c6d182'    # greenish shade
color_original = '#ae98b6'  # purple hue

plt.rcParams.update({'font.size': 18})

# Create a figure with two subplots; left wider than right.
fig, axs = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1.5]})
fig.suptitle("Tissue Segmentation IoU Comparison: Vispro vs. Original", fontsize=16)

axs[0].bar(ind - width/2, sel_iou_vispro, width, color=color_vispro, label='Vispro')
axs[0].bar(ind + width/2, sel_iou_original, width, color=color_original, label='Original image')
axs[0].set_xticks(ind)
axs[0].set_xticklabels([str(sid) for sid in sel_sample_ids], rotation=45, ha='center', fontsize=10)
axs[0].set_xlabel("Image id")
axs[0].set_ylabel("IoU")
axs[0].set_title("IoU per Selected Sample")
# Place the legend outside the plotting area:
axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, fontsize=12)
axs[0].grid(axis='y', linestyle='--', alpha=0.5)


# Right Subplot: Bar Chart for IoU Difference (Vispro - Original)
axs[1].bar(ind, sel_diff, width, color='green')
axs[1].set_xticks(ind)
axs[1].set_xticklabels([str(sid) for sid in sel_sample_ids],rotation=45, ha='center', fontsize=10)
axs[1].set_xlabel("Image id")
axs[1].set_ylabel("IoU Difference")
axs[1].set_title("Improvement by Vispro")
axs[1].axhline(0, color='black', lw=1)
axs[1].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()
