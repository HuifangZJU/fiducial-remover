import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Raw data for 20 samples:
# -------------------------------
# sample_ids = np.array([145, 41, 156, 20, 74, 147, 127, 4, 67, 100, 119, 114, 138, 90, 79, 133, 102, 84, 75, 121])
#
# # Cells in Tissue counts for each method:
# cells_in_original = np.array([70487, 84288, 75132, 49243, 66776, 33467, 61650, 19434, 54070, 57164,
#                               66005, 62655, 66118, 59234, 59662, 61331, 57709, 67734, 63701, 65808])
# cells_in_vispro1   = np.array([70343, 84285, 74268, 49897, 68047, 33555, 62425, 18002, 53658, 56747,
#                               65886, 63976, 68199, 60120, 60669, 60675, 59162, 69013, 65265, 66933])
# cells_in_vispro2   = np.array([66844, 86125, 74962, 50282, 80279, 52302, 54171, 18264, 52279, 55039,
#                               64944, 61904, 67940, 50183, 51792, 53055, 61543, 60043, 54956, 56242])
#
# # Cells out of Tissue counts for each method:
# cells_out_original = np.array([2417, 4369, 1300, 4009, 1888, 72, 261, 3143, 3586, 2328,
#                                2240, 2178, 3658, 1983, 2085, 1567, 4173, 2357, 5341, 4008])
# cells_out_vispro1   = np.array([2020, 3349, 1286, 467, 349, 54, 211, 2367, 3278, 2064,
#                                1915, 1913, 895, 1494, 1869, 1331, 1880, 1769, 4445, 2470])
# cells_out_vispro2   = np.array([603, 521, 495, 61, 283, 45, 112, 26, 3028, 1867,
#                                1463, 980, 465, 871, 1268, 635, 283, 884, 721, 114])

sample_ids = np.array([145, 41, 156, 20, 74, 147, 127, 4, 67, 100, 119, 114, 138, 90, 79, 133, 102, 84, 75, 121])

# Cells in Tissue counts for each method:
cells_in_original = np.array([70487, 84288, 75132, 49243, 66776, 33467, 61650,19434,54070, 57164,
                              66005, 62655, 66118, 59234, 59662, 61331, 57709, 67734, 63701, 65808])
cells_in_vispro2   = np.array([66844, 86125, 74962, 50282, 80279, 52302, 54171, 18264, 52279, 55039,
                              64944, 61904, 67940, 50183, 51792, 53055, 61543, 60043, 54956, 56242])

# Cells out of Tissue counts for each method:
cells_out_original = np.array([2417, 4369, 1297, 4009, 1888, 72, 261,     3143,     3586, 2328,
                               2240, 2178, 3658, 1983, 2085, 1567, 4173, 2357, 5341, 4008])
cells_out_vispro2   = np.array([207, 33, 25, 8, 164, 19, 36,     26,     658, 560,
                               864, 221, 116, 309, 475, 110, 59, 408, 270, 23])

# sample_ids = np.array([145, 41, 156, 20, 74, 147, 127, 4, 114, 138, 90, 79, 133, 102, 84, 75, 121])
#
# # Cells in Tissue counts for each method:
# cells_in_original = np.array([70487, 84288, 75132, 49243, 66776, 33467, 61650, 19434, 62655, 66118, 59234, 59662, 61331, 57709, 67734, 63701, 65808])
# cells_in_vispro1   = np.array([70343, 84285, 74268, 49897, 68047, 33555, 62425, 18002,63976, 68199, 60120, 60669, 60675, 59162, 69013, 65265, 66933])
# cells_in_vispro2   = np.array([66844, 86125, 74962, 50282, 80279, 52302, 54171, 18264, 61904, 67940, 50183, 51792, 53055, 61543, 60043, 54956, 56242])
#
# # Cells out of Tissue counts for each method:
# cells_out_original = np.array([2417, 4369, 1300, 4009, 1888, 72, 261, 3143, 2178, 3658, 1983, 2085, 1567, 4173, 2357, 5341, 4008])
# cells_out_vispro1   = np.array([2020, 3349, 1286, 467, 349, 54, 211, 2367, 1913, 895, 1494, 1869, 1331, 1880, 1769, 4445, 2470])
# cells_out_vispro2   = np.array([603, 521, 495, 61, 283, 45, 112, 26,  980, 465, 871, 1268, 635, 283, 884, 721, 114])

# -------------------------------
# For sorting, we define the improvement metric as:
# improvement = (Original out-of-tissue - Vispro2 out-of-tissue)
# (The larger the difference, the better Vispro2's reduction of out-of-tissue cells.)
# -------------------------------
improvement = cells_out_original - cells_out_vispro2
# improvement = - cells_out_vispro2

# Sort data in descending order by improvement:
sorted_indices = np.argsort(improvement)[::-1]
sel_sample_ids     = sample_ids[sorted_indices]
sel_cells_in_orig  = cells_in_original[sorted_indices]
# sel_cells_in_v1    = cells_in_vispro1[sorted_indices]
sel_cells_in_v2    = cells_in_vispro2[sorted_indices]
sel_cells_out_orig = cells_out_original[sorted_indices]
# sel_cells_out_v1   = cells_out_vispro1[sorted_indices]
sel_cells_out_v2   = cells_out_vispro2[sorted_indices]
sel_improvement    = improvement[sorted_indices]

# -------------------------------
# Plotting setup:
# -------------------------------
num_samples = len(sel_sample_ids)
ind = np.arange(num_samples)
width = 0.25  # narrower bars so all three can fit in each group

# Choose colors:
color_original = '#ae98b6'   # purple hue for Original
color_vispro1  = '#c6d182'   # olive/greenish for Vispro1
color_vispro2  = '#81c784'   # greenish for Vispro2


plt.rcParams.update({'font.size': 18})
# Create a figure with two subplots (side-by-side)
fig, axs = plt.subplots(1, 2, figsize=(18, 6), gridspec_kw={'width_ratios': [3, 3]})
fig.suptitle("Tissue Segmentation Cell Count Comparison\n(Original vs. Vispro1 vs. Vispro2)", fontsize=12)

# Left Subplot: Cells in Tissue
axs[0].bar(ind - width, sel_cells_in_v2,   width, color=color_vispro2,  label='Vispro2')
# axs[0].bar(ind,           sel_cells_in_v1,   width, color=color_vispro1,  label='Vispro1')
axs[0].bar(ind + width,  sel_cells_in_orig, width, color=color_original, label='Original image' )
axs[0].set_xticks(ind)
axs[0].set_xticklabels([str(sid) for sid in sel_sample_ids], rotation=45, ha='center', fontsize=10)
axs[0].set_xlabel("Image id")
axs[0].set_ylabel("Number of Cells")
axs[0].set_title("Detected Cells in Tissue")
axs[0].legend(loc="best", frameon=False, fontsize=12)
axs[0].grid(axis='y', linestyle='--', alpha=0.5)

# Right Subplot: Cells out of Tissue
axs[1].bar(ind - width,  sel_cells_out_v2,   width, color=color_vispro2,  label='Vispro2')
# axs[1].bar(ind,           sel_cells_out_v1,   width, color=color_vispro1,  label='Vispro1')
axs[1].bar(ind + width, sel_cells_out_orig, width, color=color_original, label='Original image' )
axs[1].set_xticks(ind)
axs[1].set_xticklabels([str(sid) for sid in sel_sample_ids], rotation=45, ha='center', fontsize=12)
axs[1].set_xlabel("Image id")
axs[1].set_ylabel("Number of Cells")
axs[1].set_title("Detected Cells out of Tissue")
axs[1].legend(loc="best", frameon=False, fontsize=12)
axs[1].axhline(0, color='black', lw=1)
axs[1].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.93])
# plt.savefig('./figures/3.png', dpi=300)
plt.show()
