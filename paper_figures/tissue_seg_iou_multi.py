import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
# --- 1. Raw metrics for 10 samples, 7 methods, 5 metrics ---
def get_metrics_results():
    raw_metrics = {
        1: {'Vispro':( 0.9131, 0),
              'Otsu (restored)':( 0.7143, 2),
              'SAM (restored)':( 0.2467, 4),
              'Tesla1 (restored)':( 0.2587, 3),
              'Otsu' :( 0.7542, 3),
              'Tesla1':( 0.2783, 4),
              'Tesla2':( 0.2816, 4),
              'Tesla3':( 0.2407, 4),
              'Bgrm' :( 0.7215, 2),
              'SAM'  :( 0.7556, 97), },
        17: { 'Vispro':( 0.9207, 1),
  'Otsu (restored)':( 0.8481, 20),
  'SAM (restored)':( 0.2824, 15),
  'Tesla1 (restored)':( 0.2894, 14),
  'Otsu' :( 0.8465, 20),
  'Tesla1':( 0.2895, 14),
  'Tesla2':( 0.3769, 15),
  'Tesla3':( 0.3508, 15),
  'Bgrm' :( 0.4745, 6),
  'SAM'  :( 0.8562, 88), },
        64: {'Vispro':( 0.9573, 0),
  'Otsu (restored)':( 0.7583, 18),
  'SAM (restored)':( 0.3832, 1),
  'Tesla1 (restored)':( 0.3990, 46),
  'Otsu' :( 0.7619, 23),
  'Tesla1':( 0.3980, 58),
  'Tesla2':( 0.6880, 2),
  'Tesla3':( 0.5371, 2),
  'Bgrm' :( 0.7002, 0),
  'SAM'  :( 0.8560, 112), },
        66: { 'Vispro':( 0.9813, 1),
  'Otsu (restored)':( 0.7321, 82),
  'SAM (restored)':( 0.9607, 0),
  'Tesla1 (restored)':( 0.9748, 32),
  'Otsu' :( 0.7313, 70),
  'Tesla1':( 0.9694, 22),
  'Tesla2':( 0.9261, 0),
  'Tesla3':( 0.9229, 0),
  'Bgrm' :( 0.9361, 24),
  'SAM'  :( 0.9107, 131), },
        67: {'Vispro':( 0.9498, 0),
  'Otsu (restored)':( 0.9337, 42),
  'SAM (restored)':( 0.9350, 2),
  'Tesla1 (restored)':( 0.9579, 61),
  'Otsu' :( 0.9294, 44),
  'Tesla1':( 0.9608, 27),
  'Tesla2':( 0.8157, 2),
  'Tesla3':( 0.8134, 2),
  'Bgrm' :( 0.7270, 9),
  'SAM'  :( 0.9350, 75),  },
        69: {'Vispro':( 0.9932, 0),
  'Otsu (restored)':( 0.9878, 1),
  'SAM (restored)':( 0.9842, 0),
  'Tesla1 (restored)':( 0.9747, 189),
  'Otsu' :( 0.9876, 3),
  'Tesla1':( 0.9693, 182),
  'Tesla2':( 0.8686, 0),
  'Tesla3':( 0.8830, 0),
  'Bgrm' :( 0.9908, 18),
  'SAM'  :( 0.9631, 114), },
        70: {'Vispro':( 0.9883, 0),
  'Otsu (restored)':( 0.9741, 13),
  'SAM (restored)':( 0.9793, 0),
  'Tesla1 (restored)':( 0.9871, 26),
  'Otsu' :( 0.9739, 16),
  'Tesla1':( 0.9866, 20),
  'Tesla2':( 0.9119, 0),
  'Tesla3':( 0.9340, 0),
  'Bgrm' :( 0.9878, 2),
  'SAM'  :( 0.9681, 63),  },
        71: { 'Vispro':( 0.9884, 1),
  'Otsu (restored)':( 0.9431, 7),
  'SAM (restored)':( 0.9711, 1),
  'Tesla1 (restored)':( 0.9751, 70),
  'Otsu' :( 0.9366, 10),
  'Tesla1':( 0.9738, 60),
  'Tesla2':( 0.8842, 1),
  'Tesla3':( 0.9255, 1),
  'Bgrm' :( 0.9806, 2),
  'SAM'  :( 0.9031, 85),  },
        73: { 'Vispro':( 0.9602, 3),
  'Otsu (restored)':( 0.9272, 37),
  'SAM (restored)':( 0.9078, 7),
  'Tesla1 (restored)':( 0.9078, 102),
  'Otsu' :( 0.9204, 42),
  'Tesla1':( 0.8774, 70),
  'Tesla2':( 0.8569, 0),
  'Tesla3':( 0.8871, 0),
  'Bgrm' :( 0.8715, 13),
  'SAM'  :( 0.8721, 170), },
        74: { 'Vispro':( 0.9914, 0),
  'Otsu (restored)':( 0.6858, 54),
  'SAM (restored)':( 0.9865, 0),
  'Tesla1 (restored)':( 0.9887, 50),
  'Otsu' :( 0.6873, 59),
  'Tesla1':( 0.9863, 23),
  'Tesla2':( 0.9232, 0),
  'Tesla3':( 0.9258, 0),
  'Bgrm' :( 0.7779, 32),
  'SAM'  :( 0.7188, 90), },
        75: { 'Vispro':( 0.9119, 2),
  'Otsu (restored)':( 0.9439, 23),
  'SAM (restored)':( 0.8972, 0),
  'Tesla1 (restored)':( 0.9688, 50),
  'Otsu' :( 0.9413, 16),
  'Tesla1':( 0.9664, 38),
  'Tesla2':( 0.8627, 0),
  'Tesla3':( 0.9084, 0),
  'Bgrm' :( 0.9132, 4),
  'SAM'  :( 0.9498, 49), },
        78: {'Vispro':( 0.8902, 2),
  'Otsu (restored)':( 0.8187, 17),
  'SAM (restored)':( 0.0276, 3),
  'Tesla1 (restored)':( 0.3040, 4),
  'Otsu' :( 0.8263, 13),
  'Tesla1':( 0.3057, 2),
  'Tesla2':( 0.3822, 5),
  'Tesla3':( 0.3720, 5),
  'Bgrm' :( 0.2622, 3),
  'SAM'  :( 0.4158, 105), },
        79: { 'Vispro':( 0.9758, 1),
  'Otsu (restored)':( 0.9571, 15),
  'SAM (restored)':( 0.9493, 1),
  'Tesla1 (restored)':( 0.9700, 36),
  'Otsu' :( 0.9562, 15),
  'Tesla1':( 0.9636, 32),
  'Tesla2':( 0.9306, 0),
  'Tesla3':( 0.9189, 0),
  'Bgrm' :( 0.9564, 8),
  'SAM'  :( 0.9359, 66),  },
        90: { 'Vispro':( 0.9763, 2),
  'Otsu (restored)':( 0.9496, 16),
  'SAM (restored)':( 0.9712, 1),
  'Tesla1 (restored)':( 0.9739, 35),
  'Otsu' :( 0.9494, 10),
  'Tesla1':( 0.9726, 34),
  'Tesla2':( 0.9423, 0),
  'Tesla3':( 0.9351, 0),
  'Bgrm' :( 0.9512, 8),
  'SAM'  :( 0.9334, 59), },
        100: { 'Vispro':( 0.9428, 0),
  'Otsu (restored)':( 0.9088, 76),
  'SAM (restored)':( 0.6896, 10),
  'Tesla1 (restored)':( 0.9455, 78),
  'Otsu' :( 0.9103, 61),
  'Tesla1':( 0.9445, 73),
  'Tesla2':( 0.7901, 1),
  'Tesla3':( 0.7964, 1),
  'Bgrm' :( 0.7220, 25),
  'SAM'  :( 0.9056, 49), },
        101: {'Vispro':( 0.0615, 6),
  'Otsu (restored)':( 0.3467, 272),
  'SAM (restored)':( 0.9617, 0),
  'Tesla1 (restored)':( 0.5901, 155),
  'Otsu' :( 0.0537, 94),
  'Tesla1':( 0.6223, 75),
  'Tesla2':( 0.5875, 0),
  'Tesla3':( 0.5297, 0),
  'Bgrm' :( 0.5261, 113),
  'SAM'  :( 0.9195, 100),  },
        110: {'Vispro':( 0.9645, 0),
  'Otsu (restored)':( 0.5302, 43),
  'SAM (restored)':( 0.9196, 2),
  'Tesla1 (restored)':( 0.9486, 113),
  'Otsu' :( 0.5379, 43),
  'Tesla1':( 0.9442, 90),
  'Tesla2':( 0.8576, 2),
  'Tesla3':( 0.8376, 2),
  'Bgrm' :( 0.6087, 29),
  'SAM'  :( 0.8455, 92), },
        141: {'Vispro':( 0.8670, 1),
  'Otsu (restored)':( 0.7900, 4),
  'SAM (restored)':( 0.0006, 2),
  'Tesla1 (restored)':( 0.4367, 146),
  'Otsu' :( 0.7491, 42),
  'Tesla1':( 0.4302, 131),
  'Tesla2':( 0.1316, 2),
  'Tesla3':( 0.1301, 2),
  'Bgrm' :( 0.2920, 1),
  'SAM'  :( 0.6108, 112), },
        149: {'Vispro':( 0.9825, 0),
  'Otsu (restored)':( 0.7278, 63),
  'SAM (restored)':( 0.0037, 0),
  'Tesla1 (restored)':( 0.9823, 34),
  'Otsu' :( 0.7271, 69),
  'Tesla1':( 0.9766, 13),
  'Tesla2':( 0.7053, 0),
  'Tesla3':( 0.7050, 0),
  'Bgrm' :( 0.9113, 34),
  'SAM'  :( 0.3803, 5), },
        161: {'Vispro':( 0.9287, 0),
  'Otsu (restored)':( 0.8591, 6),
  'SAM (restored)':( 0.0000, 1),
  'Tesla1 (restored)':( 0.0901, 106),
  'Otsu' :( 0.8705, 4),
  'Tesla1':( 0.0900, 128),
  'Tesla2':( 0.2605, 1),
  'Tesla3':( 0.3763, 1),
  'Bgrm' :( 0.8102, 2),
  'SAM'  :( 0.0929, 3), },
        163: {'Vispro':( 0.9412, 2),
  'Otsu (restored)':( 0.4898, 39),
  'SAM (restored)':( 0.9415, 0),
  'Tesla1 (restored)':( 0.9761, 105),
  'Otsu' :( 0.4913, 42),
  'Tesla1':( 0.9743, 99),
  'Tesla2':( 0.8538, 0),
  'Tesla3':( 0.8469, 0),
  'Bgrm' :( 0.6818, 23),
  'SAM'  :( 0.7786, 75),  }
    }

    metric_keys = ['IoU', 'CompDiff']

    # --- assemble nested dict in `metrics_results` format -------------
    metrics_results = {
        sid: {
            meth: dict(zip(metric_keys, vals))
            for meth, vals in raw_metrics[sid].items()
        }
        for sid in raw_metrics
    }
    return metrics_results

metrics_results = get_metrics_results()

# ─── assume metrics_results is already populated ─────────────────────────

methods  = ['Vispro','Bgrm','SAM (restored)','Otsu (restored)','Tesla1 (restored)','Tesla2','Tesla3']
plot_labels  = ['Vispro','Bgrm','SAM','Otsu','TESLA1','TESLA2','TESLA3']
metrics  = ['IoU','CompDiff']
method_to_label = dict(zip(methods, plot_labels))
# ---------- colour palette (green for Vispro, purples for others) --------
base = ['#c6d182','#e0c7e3','#eae0e9','#ae98b6','#846e89']
palette = {
    'Vispro':base[0],
    'Bgrm':mcolors.to_hex(np.array(mcolors.to_rgb(base[3])) * 0.8),
    'Tesla1 (restored)': base[3],
    'Otsu (restored)': mcolors.to_hex(np.array(mcolors.to_rgb(base[4])) * 0.8),
    'SAM (restored)': base[2],
    'Tesla2':mcolors.to_hex(np.array(mcolors.to_rgb(base[2])) * 0.8),
    'Tesla3':base[1],
}
# ---------- collect raw values ------------------------------------------
sample_ids = sorted(metrics_results.keys())
raw = {
    m: {meth: [metrics_results[s][meth][m] for s in sample_ids] for meth in methods}
    for m in metrics
}

# ---------- averages for non‑PerimRatio metrics --------------------------
avg_vals = {
    m: {meth: np.mean(raw[m][meth]) for meth in methods} for m in metrics if m != 'PerimRatio'
}

# ---------- figure / subplots -------------------------------------------
plt.rcParams.update({'font.size': 16})

# fig, axes = plt.subplots(1, len(metrics), figsize=(9, 6), sharey=False)
# bar_w = 0.7
#
# for col, metric in enumerate(metrics):                        # last subplot is PerimRatio
#     ax = axes[col]
#
#     # -----------------------------------------------
#     # special handling for CompDiff: clip at 10
#     # -----------------------------------------------
#     if metric == 'CompDiff':
#         bar_h = 0.6  # thickness of each horizontal bar
#         y_pos = np.arange(len(methods))  # vertical positions
#
#         cutoff = 10
#         vals_raw = [avg_vals[metric][m] for m in methods]
#         vals_plot = [min(v, cutoff) for v in vals_raw]
#
#         bars = ax.barh(
#             y_pos,
#             vals_plot,
#             height=bar_h,
#             color=[palette[m] for m in methods],
#             edgecolor=['black' if m == 'vispro' else 'none' for m in methods]
#         )
#
#         # annotate values that exceed the cutoff
#         for v_raw, bar in zip(vals_raw, bars):
#             if v_raw > cutoff:
#                 ax.text(
#                     cutoff - 0.2,  # x‑position near cutoff
#                     bar.get_y() + bar.get_height() / 2,  # centered on bar
#                     f'>{cutoff}',
#                     ha='right', va='center'
#                 )
#
#         ax.set_xlim(0, cutoff + 1)
#         ax.set_yticks(y_pos)
#         ax.set_yticklabels(plot_labels )
#         ax.set_title('Avg CompDiff')
#         ax.grid(axis='x', linestyle='--', alpha=0.4)
#
#
#     # -----------------------------------------------
#     # regular bar plot for IoU, HD, Solidity
#     # -----------------------------------------------
#     else:
#         heights = [avg_vals[metric][m] for m in methods]
#         y_pos = np.arange(len(methods))  # positions down the y‑axis
#
#         ax.barh(
#             y_pos,  # y‑locations
#             heights,  # bar lengths (x‑direction)
#             height=bar_w,  # bar thickness
#             color=[palette[m] for m in methods],
#             edgecolor=['black' if m == 'vispro' else 'none' for m in methods]
#         )
#
#         ax.set_yticks(y_pos)
#         ax.set_yticklabels(plot_labels )
#         ax.set_title('Avg ' + metric)
#         ax.grid(axis='x', linestyle='--', alpha=0.4)  # grid lines now horizontal axis
#
# # ── 2) PerimRatio distribution plot (violin + scatter) ───────────────────
# # ax_p = axes[-1]
# # metric = 'PerimRatio'
# # vals_all = [v for meth in methods for v in raw[metric][meth]]
# # span = max(abs(np.array(vals_all) - 1)) * 1.1
# # ax_p.set_xlim(-1, 1 + span)
# #
# # ax_p.axvline(1, color='grey', linestyle='--')
# #
# # for row, meth in enumerate(methods):
# #     data = raw[metric][meth]
# #     # violin
# #     vp = ax_p.violinplot(data, positions=[row], vert=False, showmeans=False, showmedians=False)
# #     for body in vp['bodies']:
# #         body.set_facecolor(palette[meth])
# #         body.set_edgecolor('none')
# #         body.set_alpha(0.45)
# #     # overlay sample points
# #     jitter = (np.random.rand(len(data)) - 0.5) * 0.25
# #     ax_p.scatter(
# #         data,
# #         np.full_like(data, row) + jitter,
# #         marker='o',
# #         s=np.where(np.isclose(data, 1, atol=1e-6), 160, 70),
# #         facecolor=palette[meth],
# #         edgecolor='black' if meth == 'vispro' else 'none',
# #         zorder=3
# #     )
# #
# # ax_p.set_yticks(np.arange(len(methods)))
# # ax_p.set_yticklabels(methods)
# # ax_p.set_title('PerimRatio (ideal=1)')
# # ax_p.grid(axis='x', linestyle='--', alpha=0.4)
#
# plt.tight_layout()
# plt.savefig('./figures/13.png', dpi=600)
# plt.show()

fig, axes = plt.subplots(1, len(metrics), figsize=(12, 5), sharey=False)
bar_w = 0.7  # bar width

for col, metric in enumerate(metrics):  # last subplot is PerimRatio
    ax = axes[col]

    # -----------------------------------------------
    # Sort methods based on metric
    # -----------------------------------------------
    method_scores = {m: avg_vals[metric][m] for m in methods}

    if metric in ['CompDiff', 'HD']:
        # Lower is better
        sorted_methods = sorted(method_scores, key=lambda x: -method_scores[x])
    else:
        # IoU and Solidity: higher is better
        sorted_methods = sorted(method_scores, key=lambda x: method_scores[x])

    # Always move 'Vispro' to the end
    if 'Vispro' in sorted_methods:
        sorted_methods.remove('Vispro')
        sorted_methods.append('Vispro')

    plot_order = sorted_methods
    plot_labels_sorted = [method_to_label[m] for m in plot_order]
    # plot_labels_sorted = [m.replace('(restored)', '').replace(' ', '').upper() if 'TESLA' in m or 'Otsu' in m or 'SAM' in m else m for m in plot_order]

    # Special handling for CompDiff
    if metric == 'CompDiff':
        bar_h = 0.6
        x_pos = np.arange(len(plot_order))

        cutoff = 10
        vals_raw = [avg_vals[metric][m] for m in plot_order]
        vals_plot = [min(v, cutoff) for v in vals_raw]

        bars = ax.bar(
            x_pos,
            vals_plot,
            width=bar_h,
            color=[palette[m] for m in plot_order],
            edgecolor=['black' if m == 'vispro' else 'none' for m in plot_order]
        )

        for v_raw, bar in zip(vals_raw, bars):
            if v_raw > cutoff:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    cutoff - 0.2,
                    f'>{cutoff}',
                    ha='center', va='top', fontsize=12
                )

        ax.set_ylim(0, cutoff + 1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_labels_sorted, rotation=45, ha='center')
        ax.set_title('Avg CompDiff')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    else:
        heights = [avg_vals[metric][m] for m in plot_order]
        x_pos = np.arange(len(plot_order))

        ax.bar(
            x_pos,
            heights,
            width=bar_w,
            color=[palette[m] for m in plot_order],
            edgecolor=['black' if m == 'vispro' else 'none' for m in plot_order]
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_labels_sorted, rotation=45, ha='center')
        ax.set_title('Avg ' + metric)
        ax.grid(axis='y', linestyle='--', alpha=0.4)


plt.tight_layout()
# plt.savefig('./figures/13.png', dpi=300)
plt.show()