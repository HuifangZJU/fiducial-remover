import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# 1) Define your methods and their IoU .npy paths + colours

base = ['#c6d182','#846e89','#ae98b6','#e0c7e3','#eae0e9']


inputs = [
    dict(label='Vispro',     path='./result/iou_vispros.npy',   color=base[0]),
    dict(label='U‑Net',       path='./result/iou_unets.npy',     color=base[1]),
    dict(label='10X',         path='./result/iou_10xs.npy',      color=base[2]),
    dict(label='CircleNet',   path='./result/iou_circlenets.npy',color=base[3]),
    dict(label='Cellpose',    path='./result/iou_cellposes.npy', color=mcolors.to_hex(np.array(mcolors.to_rgb(base[2])) * 0.8)),
    dict(label='Hough',       path='./result/iou_houghs.npy',    color=base[4]),
]

# 2) Define your overlap bands
ranges = {
    "=0":   (0,   0),
    "0‑10": (0,  10),
    "10‑30":(10, 30),
    "30‑50":(30, 50),
    ">50":  (50, np.inf),
}
range_labels = list(ranges.keys())

# 3) Helper to average IoU in each band, with a special case for "=0"
def mean_by_range(cov, iou):
    out = {}
    for k, (lo, hi) in ranges.items():
        if lo == hi == 0:
            mask = (cov == 0)
        else:
            mask = (cov >= lo) & (cov < hi)
        out[k] = iou[mask].mean() if mask.any() else 0.0
    return out

# 4) Load overlap percentages and sort
tissue_pct = np.load('./result/percentage.npy')
order = np.argsort(tissue_pct)
tissue_pct = tissue_pct[order]

# 5) Load each IoU array, apply sort, compute band‑means
for d in inputs:
    arr = np.load(d['path'])[order]
    d['means'] = mean_by_range(tissue_pct, arr)

# 6) Plot grouped bar chart including the “=0” bar
bar_w   = 0.12
x       = np.arange(len(range_labels))

plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(15, 6))
for i, d in enumerate(inputs):
    offset = (i - (len(inputs)-1)/2) * bar_w
    means  = [d['means'][lab] for lab in range_labels]
    ax.bar(x + offset, means,
           width=bar_w,
           color=d['color'],
           label=d['label'])

ax.set_xticks(x)
ax.set_xticklabels(range_labels)
ax.set_xlabel('Marker‑overlap band (%)')
ax.set_ylabel('Mean IoU')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.legend(frameon=False,  loc='upper right',bbox_to_anchor=(1.17, 0.98))
plt.tight_layout()
plt.savefig('./figures/7.png', dpi=300)
plt.show()
