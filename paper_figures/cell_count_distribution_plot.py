import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
# --------------------------------------------------
# data (unsorted – order is irrelevant for a distribution plot)
# --------------------------------------------------
sample_ids = np.array([145, 41, 156, 20, 74, 147, 127, 4, 67, 100, 119, 114, 138, 90, 79, 133, 102, 84, 75, 121])
cells_in  = {
    'Vispro': np.array([66844, 86125, 74962, 50282, 80279, 52302, 54171,
                          18264, 61904, 67940, 50183, 51792, 53055, 61543,
                          60043, 54956, 56242]),
    'Vispro1': np.array([70343, 84285, 74268, 49897, 68047, 33555, 62425,
                          18002, 63976, 68199, 60120, 60669, 60675, 59162,
                          69013, 65265, 66933]),
    'Original': np.array([70487, 84288, 75132, 49243, 66776, 33467, 61650,
                          19434, 62655, 66118, 59234, 59662, 61331, 57709,
                          67734, 63701, 65808]),
}
cells_out = {
    'Vispro': np.array([207, 33, 25, 8, 164, 19, 36, 26, 30, 560,
                               140, 221, 116, 309, 475, 110, 59, 408, 270, 23]),
    'Vispro1': np.array([2020, 3349,1286, 467, 349,  54, 211,2367,1913,
                            895,1494,1869,1331,1880,1769,4445,2470]),
    'Original': np.array([2417, 4369, 1297, 4009, 1888, 72, 261,     3143,     1287, 2328,
                               1422, 2178, 3658, 1983, 2085, 1567, 4173, 2357, 5341, 4008]),
}




# --------------------------------------------------
# plotting
# --------------------------------------------------

# --------------- data & colour map ---------------------------------
data_dict = cells_in                      # ONLY the in‑tissue counts
methods    = ['Vispro', 'Original']
colors     = ['#c6d182', '#ae98b6']   # match your scheme

# --------------- single violin plot --------------------------------
fig, ax = plt.subplots(1,2,figsize=(17, 4),gridspec_kw={'width_ratios': [1, 6]})
plt.rcParams.update({'font.size': 14})
# Violin (one per method)
parts = ax[0].violinplot(
    [data_dict[m] for m in methods],
    positions=np.arange(len(methods)),
    showmeans=False, showmedians=False, vert=True
)

# Colour each violin
for body, col in zip(parts['bodies'], colors):
    body.set_facecolor(col)
    body.set_edgecolor('none')
    body.set_alpha(0.45)

# Overlay individual sample points
for idx, (m, col) in enumerate(zip(methods, colors)):
    vals   = data_dict[m]
    jitter = (np.random.rand(len(vals)) - 0.5) * 0.25
    ax[0].scatter(idx + jitter, vals,
               s=60, marker='o',
               color=col,
               edgecolor='black' if m == 'Vispro2' else 'none',
               zorder=3)

# Axis decoration
ax[0].set_xticks(np.arange(len(methods)))
ax[0].set_xticklabels(methods, ha='center')
ax[0].set_ylabel('Number of detected cells')
# ax[0].set_title('#Cell distributions in tissue')
ax[0].grid(axis='y', linestyle='--', alpha=0.4)

# --- choose bar positions and thickness ---------------------------------
ind   = np.arange(len(sample_ids))      # x‑locations for groups
width = 0.45                            # bar width

# --- pick colours consistent with previous plots ------------------------


# --- out‑of‑tissue reduction for sorting -------------------------------
cells_out_original = cells_out['Original']
cells_out_vispro2  = cells_out['Vispro']
improvement = cells_out_original - cells_out_vispro2

sorted_idx = np.argsort(improvement)[::-1]

sel_sample_ids    = sample_ids[sorted_idx]
sel_cells_out_v2  = cells_out['Vispro'][sorted_idx]
# sel_cells_out_v1  = cells_out['Vispro1'][sorted_idx]
sel_cells_out_orig= cells_out['Original'][sorted_idx]

# --- plot on ax[1] ------------------------------------------------------
# ax[1].bar(ind - width, sel_cells_out_v2,  width,
#           color=color_vispro2,  label='Vispro')
# # ax[1].bar(ind,         sel_cells_out_v1,  width,
# #           color=color_vispro1,  label='Vispro1')
# ax[1].bar(ind , sel_cells_out_orig,width,
#           color=color_original, label='Original')
#
# ax[1].set_xticks(ind)
# ax[1].set_xticklabels(sel_sample_ids, rotation=45, ha='center')
# ax[1].set_xlabel('Sample ID')
# ax[1].set_ylabel('Out‑of‑tissue cell count')
# ax[1].set_title('#Cell outside tissue per sample')
# ax[1].legend(frameon=False)
# ax[1].grid(axis='y', linestyle='--', alpha=0.5)

# Your bar plots
ax[1].bar(ind - width, sel_cells_out_v2, width, color=colors[0], label='Vispro')
ax[1].bar(ind, sel_cells_out_orig, width, color=colors[1], label='Original')

# Set x-ticks and labels
ax[1].set_xticks(ind)
ax[1].set_xticklabels(sel_sample_ids, ha='center', fontsize=12)
ax[1].set_xlabel('Sample ID', fontsize=12)
ax[1].set_ylabel('Out‑of‑tissue cell count', fontsize=12)
ax[1].set_title('Count of cells outside tissue regions per sample')
ax[1].legend(frameon=False)
ax[1].grid(axis='y', linestyle='--', alpha=0.5)

# Set y-axis to log scale
# ax[1].set_yscale('log')

# Add number annotations
# for bar in ax[1].patches:
#     height = bar.get_height()
#     if height > 0:
#         ax[1].annotate(f'{int(height)}',
#                        xy=(bar.get_x() + bar.get_width() / 2, height),
#                        xytext=(0, 1),  # very tight
#                        textcoords='offset points',
#                        ha='center', va='bottom', fontsize=10)

# Adjust y-limits to leave room for text
ymin, ymax = ax[1].get_ylim()
ax[1].set_ylim(ymin, ymax)  # 10% higher


extent = ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
expand = 0.7  # For example, expand by 10% on each side
new_extent = mtransforms.Bbox.from_extents(
    extent.x0 - expand, extent.y0 - expand,
    extent.x1 + expand, extent.y1 + expand
)

fig.savefig('/home/huifang/workspace/code/fiducial_remover/paper_figures/figures/6.png', bbox_inches=new_extent,dpi=600)
# plt.savefig('./figures/6.png', dpi=300)
plt.tight_layout()
plt.show()