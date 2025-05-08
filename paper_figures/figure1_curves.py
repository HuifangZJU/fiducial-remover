import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Transform
from matplotlib.ticker import FuncFormatter, FixedLocator
from matplotlib.scale import ScaleBase
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.transforms import Transform
from matplotlib.ticker import FuncFormatter, FixedLocator
from matplotlib.scale import ScaleBase, register_scale

# Custom Transform for Y-Axis
class CustomYTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True

    def transform_non_affine(self, y):
        # Map each specified interval to an equal portion of the y-axis
        return np.piecewise(
            y,
            [y < 0.5, (y >= 0.5) & (y < 0.8), (y >= 0.8) & (y < 0.9), y >= 0.9],
            [
                lambda y: y / 0.5 * 0.25,            # Map [0, 0.5] to [0, 0.25]
                lambda y: 0.25 + (y - 0.5) / 0.3 * 0.25, # Map [0.5, 0.8] to [0.25, 0.5]
                lambda y: 0.5 + (y - 0.8) / 0.1 * 0.25, # Map [0.8, 0.9] to [0.5, 0.75]
                lambda y: 0.75 + (y - 0.9) / 0.1 * 0.25 # Map [0.9, 1.0] to [0.75, 1.0]
            ]
        )

    def inverted(self):
        return InvertedCustomYTransform()

class InvertedCustomYTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True

    def transform_non_affine(self, y):
        # Inverse of the above transformation for accurate axis representation
        return np.piecewise(
            y,
            [y < 0.25, (y >= 0.25) & (y < 0.5), (y >= 0.5) & (y < 0.75), y >= 0.75],
            [
                lambda y: y * 0.5 / 0.25,                # Map [0, 0.25] back to [0, 0.5]
                lambda y: 0.5 + (y - 0.25) * 0.3 / 0.25, # Map [0.25, 0.5] back to [0.5, 0.8]
                lambda y: 0.8 + (y - 0.5) * 0.1 / 0.25,  # Map [0.5, 0.75] back to [0.8, 0.9]
                lambda y: 0.9 + (y - 0.75) * 0.1 / 0.25  # Map [0.75, 1.0] back to [0.9, 1.0]
            ]
        )

# Custom Scale for Y-Axis
class CustomYScale(ScaleBase):
    name = 'custom_y'

    def get_transform(self):
        return CustomYTransform()

    def set_default_locators_and_formatters(self, axis, _=None):
        axis.set_major_locator(FixedLocator([0, 0.5, 0.8, 0.9, 1.0]))
        axis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1f}'))

# Custom Transform for X-Axis with Uniform Lengths for Intervals
class CustomXTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True

    def transform_non_affine(self, x):
        # Map each specified interval to an equal portion of the x-axis
        return np.piecewise(
            x,
            [
                x<0,
                # x == 0,  # Exact 0 point
                (x >= 0) & (x <= 10),  # [0, 10]
                (x > 10) & (x <= 30),  # [10, 30]
                (x > 30) & (x <= 50),  # [30, 50]
                x > 50  # [50, 100]
            ],
            [
                lambda x: x,
                # lambda x: 0,  # Map 0 to 0
                lambda x: 0.05 + (x / 10) * 0.2,  # Map [0, 10] to [0.2, 0.4]
                lambda x: 0.25 + ((x - 10) / 20) * 0.2,  # Map [10, 30] to [0.4, 0.6]
                lambda x: 0.45 + ((x - 30) / 20) * 0.2,  # Map [30, 50] to [0.6, 0.8]
                lambda x: 0.65 + ((x - 50) / 50) * 0.2  # Map [50, 100] to [0.8, 1.0]
            ]
        )

    def inverted(self):
        return InvertedCustomXTransform()

class InvertedCustomXTransform(Transform):
    input_dims = 1
    output_dims = 1
    is_separable = True

    def transform_non_affine(self, x):
        # Inverse transformation for accurate axis representation
        return np.piecewise(
            x,
            [
                x<0,
                # x == 0,                      # Map [0] to [0]
                (x >=0 ) & (x <= 0.3),        # Map (0, 0.4] back to (0, 10]
                (x > 0.3) & (x <= 0.5),      # Map (0.4, 0.6] back to [10, 30]
                (x > 0.5) & (x <= 0.7),      # Map (0.6, 0.8] back to [30, 50]
                x > 0.7                       # Map (0.8, 1.0] back to [50, 100]
            ],
            [
                lambda x: x,
                # lambda x: 0,                                 # Map 0 to 0
                lambda x: 10 * (x - 0.05) / 0.2,             # Map (0.2, 0.4] back to (0, 10]
                lambda x: 10 + (x - 0.25) * 20 / 0.2,        # Map (0.4, 0.6] back to [10, 30]
                lambda x: 30 + (x - 0.45) * 20 / 0.2,        # Map (0.6, 0.8] back to [30, 50]
                lambda x: 50 + (x - 0.65) * 50 / 0.2         # Map (0.8, 1.0] back to [50, 100]
            ]
        )


# Custom Scale for X-Axis
class CustomXScale(ScaleBase):
    name = 'custom_x'

    def get_transform(self):
        return CustomXTransform()

    def set_default_locators_and_formatters(self, axis, _=None):
        # Set major ticks at the endpoints of each interval
        axis.set_major_locator(FixedLocator([0, 10, 30, 50, 100]))
        axis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}'))
# Function to calculate average IoU within specified percentage ranges
def average_iou_by_range(tissue_percentage, iou, ranges):
    avg_iou = {}
    for label, (low, high) in ranges.items():
        if high ==0 :
            mask = tissue_percentage ==0
            if np.any(mask):  # Ensure there are values in this range
                avg_iou[label] = np.mean(iou[mask])
            else:
                avg_iou[label] = np.nan  # Handle empty ranges gracefully
        else:
            mask = (tissue_percentage > low) & (tissue_percentage <= high)
            if np.any(mask):  # Ensure there are values in this range
                avg_iou[label] = np.mean(iou[mask])
            else:
                avg_iou[label] = np.nan  # Handle empty ranges gracefully
    return avg_iou

register_scale(CustomYScale)

register_scale(CustomXScale)

# ------------------------------------------------------------------
base = ['#c6d182','#e0c7e3','#ae98b6','#eae0e9','#846e89']
palette = [base[0],
           base[4],
            mcolors.to_hex(np.array(mcolors.to_rgb(base[2])) * 0.8),
            base[2],
           mcolors.to_hex(np.array(mcolors.to_rgb(base[3])) * 0.8),
           base[3],
            base[1]
           ]

inputs = [
    dict(label='Vispro',
         path='./result/iou_vispros.npy',
         color=palette[0],      # deep indigo
         marker='X'),

    dict(label='Baseline UNet',
         path='./result/iou_unets.npy',
         color=palette[1],      # rich teal‑green
         marker='v'),

    dict(label='10x pipeline',
         path='./result/iou_10xs.npy',
         color=palette[2],      # warm orange
         marker='^'),

    dict(label='CircleNet',
             path='./result/iou_circlenets.npy',
             color=palette[3],      # fresh olive‑green
             marker='o'),
    dict(label='Cellpose',
             path='./result/iou_cellposes.npy',
             color=palette[4],      # muted violet
             marker='D'),
    dict(label='Hough transform',
         path='./result/iou_houghs.npy',
         color=palette[5],      # vivid magenta
         marker='s'),


]


# ------------------------------------------------------------------
# 2)  load percentage once + sort index
# ------------------------------------------------------------------
tissue_percentage = np.load('./result/percentage.npy')
sort_idx = np.argsort(tissue_percentage)
tissue_percentage = tissue_percentage[sort_idx]


# ------------------------------------------------------------------
# 3)  helper to compute mean IoU in five coverage bands
# ------------------------------------------------------------------
ranges = {"=0":(0,0), "0‑10":(0,10), "10‑30":(10,30),
          "30‑50":(30,50), ">50":(50, np.inf)}
x_pos  = [0, 5, 20, 40, 75]

def mean_by_range(cov, iou):
    out={}
    for k,(lo,hi) in ranges.items():
        if lo==hi:
            m = (cov==lo)
        else:
            m = (cov>=lo)&(cov<hi)
        out[k]=iou[m].mean() if m.any() else np.nan
    return out

# ------------------------------------------------------------------
# 4)  load IoU arrays, apply sorting, store means
# ------------------------------------------------------------------
for d in inputs:
    arr          = np.load(d['path'])[sort_idx]
    d['iou']     = arr
    d['means']   = mean_by_range(tissue_percentage, arr)

# ------------------------------------------------------------------
# 5)  plotting (everything else below is your original code)
# ------------------------------------------------------------------
plt.figure(figsize=(12.7, 5.5))
plt.grid(True, axis='x')

# scatter + mean curve per method
for d in inputs:
    plt.scatter(tissue_percentage, d['iou'],
                label=d['label'], color=d['color'],
                marker=d['marker'], s=40)



    plt.plot(x_pos,
             [d['means'][r] for r in ranges],
             color=d['color'], marker=d['marker'],
             markersize=8, linewidth=2)

# ----------------- your axis / label / legend block ------------------
ax = plt.gca()

ax.text(-0.001, 1.013, "   No overlap", fontsize=14, color='black')
ax.text(2.7, 1.013, "Slight overlap", fontsize=14, color='black')
ax.text(12.04, 1.013, "Moderate overlap", fontsize=14, color='black')
ax.text(34.04, 1.013, "High overlap", fontsize=14, color='black')
ax.text(55.04, 1.013, "Very high overlap", fontsize=14, color='black')

ax.set_xscale('custom_x')
ax.set_xlim(-0.01, 105)
ax.set_xticks([0, 10, 30, 50, 100])
ax.set_xticklabels(["0%", "10%", "30%", "50%", "100%"], fontsize=14)

ax.set_yticks([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
ax.set_yticklabels(['0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'], fontsize=14)
ax.set_ylim(0, 1)

plt.xlabel('Marker overlap proportion with tissue area', fontsize=18)
plt.ylabel('IoU', fontsize=18)
ax.legend(loc="lower right", fontsize=12, frameon=True, bbox_to_anchor=(1.0, -0.01))

plt.tight_layout()
plt.savefig('./figures/4.png', dpi=300)
plt.show()
