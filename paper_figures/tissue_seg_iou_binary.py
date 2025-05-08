import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
# --- 1. Raw metrics for 10 samples, 7 methods, 5 metrics ---
def get_metrics_results():
    raw_metrics = {
        1: {'Vispro':( 0.9131, 430.12, 0.7992),
              'Otsu (restored)':( 0.7268, 590.76, 1.5521),
              'SAM (restored)':( 0.2467, 737.84, 0.1976),
              'Tesla1(restored)':( 0.2587, 1160.63, 0.4896),
              'SAM'  :( 0.2480, 737.84, 0.1978),
              'Otsu' :( 0.4289, 599.29, 9.4453),
              'Tesla1':( 0.2783, 1160.27, 0.2653),
              'Tesla2':( 0.2816, 361.67, 2.0467),
              'Tesla3':( 0.2407, 406.28, 1.8288),
              'Bgrm' :( 0.7214, 568.82, 0.6392)},
        17: { 'Vispro':( 0.9204, 81.34, 0.9855),
              'Otsu (restored)':( 0.8508, 540.70, 1.7323),
              'SAM (restored)':( 0.2824, 804.58, 0.1472),
              'Tesla1(restored)':( 0.2894, 783.30, 0.2087),
              'SAM'  :( 0.2668, 823.65, 0.1343),
              'Otsu' :( 0.8492, 591.71, 1.7436),
              'Tesla1':( 0.2895, 783.30, 0.2068),
              'Tesla2':( 0.3769, 372.52, 0.5871),
              'Tesla3':( 0.3508, 335.00, 0.8746),
              'Bgrm' :( 0.4772, 408.98, 0.6647)},
        64: {'Vispro':( 0.9567, 78.11, 1.0047),
              'Otsu (restored)':( 0.7587, 543.77, 2.3207),
              'SAM (restored)':( 0.3833, 656.59, 0.3785),
              'Tesla1(restored)':( 0.3990, 602.85, 0.9075),
              'SAM'  :( 0.3831, 658.29, 0.3762),
              'Otsu' :( 0.7293, 737.91, 7.1696),
              'Tesla1':( 0.3980, 602.85, 1.0207),
              'Tesla2':( 0.6880, 292.16, 1.3714),
              'Tesla3':( 0.5371, 301.32, 1.5348),
              'Bgrm' :( 0.7004, 657.92, 0.8579)},
        66: {'Vispro':( 0.9813, 69.35, 0.9231),
              'Otsu (restored)':( 0.7499, 709.74, 7.2272),
              'SAM (restored)':( 0.9608, 49.09, 0.9359),
              'Tesla1(restored)':( 0.9748, 275.69, 1.9369),
              'SAM'  :( 0.9607, 55.17, 0.9244),
              'Otsu' :( 0.7158, 974.48, 13.4152),
              'Tesla1':( 0.9694, 275.69, 1.9301),
              'Tesla2':( 0.9261, 76.90, 0.8475),
              'Tesla3':( 0.9229, 84.72, 0.8812),
              'Bgrm' :( 0.9378, 168.43, 1.7273),},
        67: { 'Vispro':( 0.9498, 149.82, 0.9131),
              'Otsu (restored)':( 0.9291, 598.00, 2.9466),
              'SAM (restored)':( 0.9352, 181.53, 0.8748),
              'Tesla1(restored)':( 0.9579, 110.14, 2.0380),
              'SAM'  :( 0.9440, 178.94, 0.7764),
              'Otsu' :( 0.8941, 597.40, 5.6693),
              'Tesla1':( 0.9608, 110.14, 2.1494),
              'Tesla2':( 0.8157, 287.00, 0.6423),
              'Tesla3':( 0.8134, 288.00, 0.7523),
              'Bgrm' :( 0.7270, 462.36, 1.0719),},
        69: { 'Vispro':( 0.9932, 12.65, 0.9842),
              'Otsu (restored)':( 0.9873, 548.73, 1.2951),
              'SAM (restored)':( 0.9842, 17.03, 0.9618),
              'Tesla1(restored)':( 0.9747, 133.63, 3.0353),
              'SAM'  :( 0.9836, 18.60, 0.9651),
              'Otsu' :( 0.9868, 562.39, 1.3762),
              'Tesla1':( 0.9693, 123.26, 3.2087),
              'Tesla2':( 0.8686, 161.14, 0.8341),
              'Tesla3':( 0.8830, 158.50, 0.9003),
              'Bgrm' :( 0.9904, 31.30, 1.4671),},
        70: { 'Vispro':( 0.9882, 68.01, 0.9507),
              'Otsu (restored)':( 0.9737, 508.32, 1.2152),
              'SAM (restored)':( 0.9793, 20.81, 0.9496),
              'Tesla1(restored)':( 0.9871, 34.00, 1.4546),
              'SAM'  :( 0.9786, 21.95, 0.9507),
              'Otsu' :( 0.9222, 534.08, 6.3332),
              'Tesla1':( 0.9866, 34.00, 1.4104),
              'Tesla2':( 0.9119, 168.00, 0.8930),
              'Tesla3':( 0.9340, 180.00, 0.9611),
              'Bgrm' :( 0.9875, 33.24, 1.2015),},
        71: { 'Vispro':( 0.9882, 59.14, 1.0212),
              'Otsu (restored)':( 0.9346, 763.70, 2.8472),
              'SAM (restored)':( 0.9711, 170.47, 0.8827),
              'Tesla1(restored)':( 0.9751, 153.44, 1.7124),
              'SAM'  :( 0.9691, 172.00, 0.8720),
              'Otsu' :( 0.8073, 851.96, 10.4190),
              'Tesla1':( 0.9738, 151.75, 1.7070),
              'Tesla2':( 0.8842, 223.29, 0.9354),
              'Tesla3':( 0.9255, 213.06, 0.9802),
              'Bgrm' :( 0.9803, 159.62, 1.1487),},
        73: { 'Vispro':( 0.9598, 152.61, 1.0630),
              'Otsu (restored)':( 0.9242, 396.09, 2.3203),
              'SAM (restored)':( 0.9065, 189.77, 1.2676),
              'Tesla1(restored)':( 0.9078, 216.84, 2.4141),
              'SAM'  :( 0.9113, 192.35, 1.2448),
              'Otsu' :( 0.8477, 562.63, 5.9511),
              'Tesla1':( 0.8774, 216.84, 2.1329),
              'Tesla2':( 0.8569, 192.63, 0.5833),
              'Tesla3':( 0.8871, 247.29, 0.8210),
              'Bgrm' :( 0.8709, 208.17, 1.0667),},
        74: { 'Vispro':( 0.9914, 13.34, 0.9606),
              'Otsu (restored)':( 0.6889, 458.84, 2.6428),
              'SAM (restored)':( 0.9865, 20.62, 0.9529),
              'Tesla1(restored)':( 0.9887, 16.12, 1.7130),
              'SAM'  :( 0.9858, 19.24, 0.9577),
              'Otsu' :( 0.6437, 458.84, 7.4982),
              'Tesla1':( 0.9863, 25.32, 1.6436),
              'Tesla2':( 0.9232, 117.11, 0.8995),
              'Tesla3':( 0.9258, 117.27, 0.9566),
              'Bgrm' :( 0.7766, 463.04, 2.0011),},
        75: { 'Vispro':( 0.9117, 255.00, 0.9251),
              'Otsu (restored)':( 0.9367, 660.66, 2.9877),
              'SAM (restored)':( 0.8972, 396.71, 0.6988),
              'Tesla1(restored)':( 0.9688, 94.58, 1.8131),
              'SAM'  :( 0.8953, 397.47, 0.6745),
              'Otsu' :( 0.8709, 660.08, 7.1129),
              'Tesla1':( 0.9664, 108.46, 1.8340),
              'Tesla2':( 0.8627, 177.34, 0.7759),
              'Tesla3':( 0.9084, 136.62, 0.9052),
              'Bgrm' :( 0.9132, 386.41, 0.9076),},
        78: {'Vispro':( 0.8901, 226.82, 0.9909),
              'Otsu (restored)':( 0.8157, 779.91, 1.9717),
              'SAM (restored)':( 0.0287, 947.53, 1.3970),
              'Tesla1(restored)':( 0.3040, 1118.43, 0.5439),
              'SAM'  :( 0.1051, 832.29, 1.2403),
              'Otsu' :( 0.6852, 834.06, 6.1440),
              'Tesla1':( 0.3057, 1118.52, 0.4997),
              'Tesla2':( 0.3822, 337.38, 0.9845),
              'Tesla3':( 0.3720, 331.45, 1.4085),
              'Bgrm' :( 0.2623, 1011.61, 0.2591),},
        79: { 'Vispro':( 0.9757, 59.48, 0.8141),
              'Otsu (restored)':( 0.9544, 650.25, 2.2137),
              'SAM (restored)':( 0.9499, 126.37, 0.8484),
              'Tesla1(restored)':( 0.9700, 85.70, 1.8274),
              'SAM'  :( 0.9415, 127.61, 0.7293),
              'Otsu' :( 0.8931, 649.80, 6.0970),
              'Tesla1':( 0.9636, 107.87, 2.0306),
              'Tesla2':( 0.9306, 116.16, 0.7372),
              'Tesla3':( 0.9189, 118.00, 0.7335),
              'Bgrm' :( 0.9560, 109.66, 1.2967),},
        90: {'Vispro':( 0.9760, 132.19, 0.9818),
              'Otsu (restored)':( 0.9435, 610.38, 3.0200),
              'SAM (restored)':( 0.9712, 48.75, 0.8859),
              'Tesla1(restored)':( 0.9739, 65.97, 2.2195),
              'SAM'  :( 0.9659, 83.00, 0.8592),
              'Otsu' :( 0.8671, 609.89, 7.8770),
              'Tesla1':( 0.9726, 105.00, 2.1589),
              'Tesla2':( 0.9423, 130.77, 0.9366),
              'Tesla3':( 0.9351, 129.63, 0.9735),
              'Bgrm' :( 0.9510, 165.50, 1.2019),},
        100: {'Vispro':( 0.9428, 164.00, 0.9895),
              'Otsu (restored)':( 0.9058, 519.40, 3.8274),
              'SAM (restored)':( 0.6916, 237.99, 1.5761),
              'Tesla1(restored)':( 0.9455, 93.49, 2.8686),
              'SAM'  :( 0.7276, 278.00, 2.1293),
              'Otsu' :( 0.8808, 519.40, 6.4101),
              'Tesla1':( 0.9445, 105.39, 2.9429),
              'Tesla2':( 0.7901, 266.00, 0.7469),
              'Tesla3':( 0.7964, 267.00, 0.8538),
              'Bgrm' :( 0.7227, 477.02, 1.5157),},
        101: {'Vispro':( 0.0618, 1261.42, 0.5320),
              'Otsu (restored)':( 0.3896, 772.77, 20.9692),
              'SAM (restored)':( 0.9617, 84.05, 1.1203),
              'Tesla1(restored)':( 0.5901, 771.51, 8.0466),
              'SAM'  :( 0.9458, 140.36, 1.2372),
              'Otsu' :( 0.0854, 772.77, 14.6596),
              'Tesla1':( 0.6223, 771.51, 5.5210),
              'Tesla2':( 0.5875, 521.51, 0.8738),
              'Tesla3':( 0.5297, 581.33, 0.9235),
              'Bgrm' :( 0.5344, 432.00, 3.5749),},
        110: { 'Vispro':( 0.9642, 103.16, 1.0205),
              'Otsu (restored)':( 0.5475, 560.86, 4.0552),
              'SAM (restored)':( 0.9199, 58.19, 1.0046),
              'Tesla1(restored)':( 0.9486, 214.66, 2.4591),
              'SAM'  :( 0.4552, 524.39, 0.7015),
              'Otsu' :( 0.5471, 847.68, 9.8080),
              'Tesla1':( 0.9442, 250.13, 2.1825),
              'Tesla2':( 0.8576, 189.01, 0.8267),
              'Tesla3':( 0.8376, 202.00, 0.8668),
              'Bgrm' :( 0.6116, 314.40, 1.8971),},
        141: { 'Vispro':( 0.8650, 50.21, 1.0207),
              'Otsu (restored)':( 0.7824, 534.90, 1.2839),
              'SAM (restored)':( 0.0006, 592.66, 1.0213),
              'Tesla1(restored)':( 0.4367, 857.92, 1.1523),
              'SAM'  :( 0.0011, 575.26, 1.4838),
              'Otsu' :( 0.5503, 566.62, 3.5752),
              'Tesla1':( 0.4302, 803.65, 1.0910),
              'Tesla2':( 0.1316, 473.79, 0.8627),
              'Tesla3':( 0.1301, 348.57, 1.4561),
              'Bgrm' :( 0.2921, 1662.03, 0.1772),},
        149: {'Vispro':( 0.9825, 36.67, 0.9413),
              'Otsu (restored)':( 0.7422, 614.72, 5.7887),
              'SAM (restored)':( 0.0037, 786.08, 0.0371),
              'Tesla1(restored)':( 0.9823, 52.39, 1.4791),
              'SAM'  :( 0.0038, 786.08, 0.0375),
              'Otsu' :( 0.7378, 614.72, 7.2858),
              'Tesla1':( 0.9766, 51.42, 1.6606),
              'Tesla2':( 0.7053, 270.65, 0.5270),
              'Tesla3':( 0.7050, 266.72, 0.5324),
              'Bgrm' :( 0.9096, 145.00, 1.5382),},
        161: { 'Vispro':( 0.9287, 34.18, 0.9699),
              'Otsu (restored)':( 0.8592, 739.42, 1.5583),
              'SAM (restored)':( 0.0000, 973.59, 0.0141),
              'Tesla1(restored)':( 0.0901, 810.25, 2.3994),
              'SAM'  :( 0.0000, 973.69, 0.0141),
              'Otsu' :( 0.7610, 789.23, 5.4789),
              'Tesla1':( 0.0900, 810.25, 2.8516),
              'Tesla2':( 0.2605, 453.26, 0.9955),
              'Tesla3':( 0.3763, 468.18, 1.0493),
              'Bgrm' :( 0.8102, 438.55, 0.9126),},
        163: {'Vispro':( 0.9420, 93.02, 1.2524),
              'Otsu (restored)':( 0.5092, 755.31, 4.7466),
              'SAM (restored)':( 0.9416, 80.89, 0.9796),
              'Tesla1(restored)':( 0.9761, 51.24, 2.3207),
              'SAM'  :( 0.9061, 82.04, 1.1866),
              'Otsu' :( 0.5154, 799.04, 6.5431),
              'Tesla1':( 0.9743, 50.60, 2.3840),
              'Tesla2':( 0.8538, 199.00, 0.7991),
              'Tesla3':( 0.8469, 212.00, 0.8164),
              'Bgrm' :( 0.6851, 343.07, 2.0291),}
    }

    metric_keys = ['IoU', 'HD',  'PerimRatio']

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
methods  = ['Tesla3', 'Tesla2', 'SAM (restored)', 'Otsu (restored)', 'Tesla1(restored)', 'Bgrm', 'Vispro']
plot_labels  = ['TESLA3', 'TESLA2', 'SAM', 'Otsu', 'TESLA1', 'Bgrm', 'Vispro']
method_to_label = dict(zip(methods, plot_labels))

# methods  = ['Vispro','Bgrm','Tesla1(restored)','Tesla1','Otsu (restored)','Otsu','SAM (restored)','SAM','Tesla2','Tesla3']
metrics  = ['IoU','HD','PerimRatio']
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


# ---------- colour palette (green for Vispro, purples for others) --------
base = ['#c6d182','#e0c7e3','#eae0e9','#ae98b6','#846e89']



palette = {
'Vispro':base[0],
'Bgrm':mcolors.to_hex(np.array(mcolors.to_rgb(base[3])) * 0.8),
'Tesla1(restored)': base[3],
'Otsu (restored)': mcolors.to_hex(np.array(mcolors.to_rgb(base[4])) * 0.8),
'SAM (restored)': base[2],
'Tesla2':mcolors.to_hex(np.array(mcolors.to_rgb(base[2])) * 0.8),
'Tesla3':base[1],
}


# ---------- figure / subplots -------------------------------------------
plt.rcParams.update({'font.size': 16})
# fig, axes = plt.subplots(1, len(metrics), figsize=(14, 6), sharey=False)
# bar_w = 0.7
#
# for col, metric in enumerate(metrics[:-1]):                        # last subplot is PerimRatio
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
#         ax.set_yticklabels(plot_labels)
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
#         ax.set_yticklabels(plot_labels)
#         ax.set_title('Avg ' + metric)
#         ax.grid(axis='x', linestyle='--', alpha=0.4)  # grid lines now horizontal axis
#
# # ── 2) PerimRatio distribution plot (violin + scatter) ───────────────────
# ax_p = axes[-1]
# metric = 'PerimRatio'
# vals_all = [v for meth in methods for v in raw[metric][meth]]
# span = max(abs(np.array(vals_all) - 1)) * 1.1
# ax_p.set_xlim(-1, 1 + span)
#
# ax_p.axvline(1, color='grey', linestyle='--')
#
# for row, meth in enumerate(methods):
#     data = raw[metric][meth]
#     # violin
#     vp = ax_p.violinplot(data, positions=[row], vert=False, showmeans=False, showmedians=False)
#     for body in vp['bodies']:
#         body.set_facecolor(palette[meth])
#         body.set_edgecolor('none')
#         body.set_alpha(0.45)
#     # overlay sample points
#     jitter = (np.random.rand(len(data)) - 0.5) * 0.25
#     ax_p.scatter(
#         data,
#         np.full_like(data, row) + jitter,
#         marker='o',
#         s=np.where(np.isclose(data, 1, atol=1e-6), 160, 70),
#         facecolor=palette[meth],
#         edgecolor='black' if meth == 'vispro' else 'none',
#         zorder=3
#     )
#
# ax_p.set_yticks(np.arange(len(methods)))
# ax_p.set_yticklabels(plot_labels)
# ax_p.set_title('PerimRatio (ideal=1)')
# ax_p.grid(axis='x', linestyle='--', alpha=0.4)
#
# plt.tight_layout()
# # plt.savefig('./figures/5.png', dpi=300)
# plt.show()

fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5), sharey=False)
bar_w = 0.7  # bar width

for col, metric in enumerate(metrics[:-1]):  # last subplot is PerimRatio
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
                    ha='center', va='top', fontsize=8
                )

        ax.set_ylim(0, cutoff + 1)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_labels_sorted, rotation=45, ha='right')
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
        ax.set_xticklabels(plot_labels_sorted, rotation=45, ha='right')
        ax.set_title('Avg ' + metric)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

# ── PerimRatio Distribution ─────────────────────────────────────
ax_p = axes[-1]
metric = 'PerimRatio'

# For PerimRatio: sort methods based on the maximum value in each group
method_max_perim = {m: max(raw[metric][m]) for m in methods}

sorted_methods_perim = sorted(method_max_perim, key=lambda x: -method_max_perim[x])
if 'Vispro' in sorted_methods_perim:
    sorted_methods_perim.remove('Vispro')
    sorted_methods_perim.append('Vispro')

plot_labels_perim = [method_to_label[m] for m in plot_order]
# plot_labels_perim = [m.replace('(restored)', '').replace(' ', '').upper() if 'TESLA' in m or 'Otsu' in m or 'SAM' in m else m for m in sorted_methods_perim]

vals_all = [v for meth in methods for v in raw[metric][meth]]
span = max(abs(np.array(vals_all) - 1)) * 1.1
ax_p.set_ylim(-1, 1 + span)

ax_p.axhline(1, color='grey', linestyle='--')

for row, meth in enumerate(sorted_methods_perim):
    data = raw[metric][meth]
    vp = ax_p.violinplot(
        data,
        positions=[row],
        vert=True,
        showmeans=False,
        showmedians=False
    )
    for body in vp['bodies']:
        body.set_facecolor(palette[meth])
        body.set_edgecolor('none')
        body.set_alpha(0.45)

    # Optional: style or hide the center bar
    if 'cbars' in vp:
        vp['cbars'].set_color(palette[meth])  # match body color
        vp['cbars'].set_alpha(0.45)
    # Hide the min/max caps (horizontal stops)
    if 'cmins' in vp and 'cmaxes' in vp:
        vp['cmins'].set_color(palette[meth])
        vp['cmaxes'].set_color(palette[meth])

    jitter = (np.random.rand(len(data)) - 0.5) * 0.25
    ax_p.scatter(
        np.full_like(data, row) + jitter,
        data,
        marker='o',
        s=np.where(np.isclose(data, 1, atol=1e-6), 160, 70),
        facecolor=palette[meth],
        edgecolor='black' if meth == 'vispro' else 'none',
        zorder=3
    )

ax_p.set_xticks(np.arange(len(sorted_methods_perim)))
ax_p.set_xticklabels(plot_labels_perim, rotation=45, ha='right')
ax_p.set_title('PerimRatio')
ax_p.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
# plt.savefig('./figures/5.png', dpi=300)
plt.show()