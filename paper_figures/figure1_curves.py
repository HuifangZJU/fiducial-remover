import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Transform
from matplotlib.ticker import FuncFormatter, FixedLocator
from matplotlib.scale import ScaleBase
import numpy as np
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


register_scale(CustomYScale)

register_scale(CustomXScale)

tissue_percentage = np.array(np.load('./result/percentage.npy'))
iou1 = np.array(np.load('./result/iou_vispros.npy'))
iou2 = np.array(np.load('./result/iou_unets.npy'))
iou3 = np.array(np.load('./result/iou_10xs.npy'))
# Sort the tissue_percentage and IoUs
sorted_indices = np.argsort(tissue_percentage)
tissue_percentage = tissue_percentage[sorted_indices]
iou1 = iou1[sorted_indices]
iou2 = iou2[sorted_indices]
iou3 = iou3[sorted_indices]


ranges = {
    "=0": (0, 0),
    "0-10": (0, 10),
    "10-30": (10, 30),
    "30-50": (30, 50),
    ">50": (50, np.inf)
}

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

# Calculate average IoUs for each curve and range
avg_iou1 = average_iou_by_range(tissue_percentage, iou1, ranges)
avg_iou2 = average_iou_by_range(tissue_percentage, iou2, ranges)
avg_iou3 = average_iou_by_range(tissue_percentage, iou3, ranges)

# Prepare data for plotting
range_labels = list(ranges.keys())
x_positions = [0, 5, 20, 40, 75]  # Approximate x positions for ranges

plt.figure(figsize=(10, 6))
plt.grid(True,axis='x')
# plt.scatter(tissue_percentage, iou1, label='Vispro', color='#e6b532', marker='o', alpha=0.4)
# plt.scatter(tissue_percentage, iou2, label='UNet', color='#d75425', marker='s', alpha=0.4)
# plt.scatter(tissue_percentage, iou3, label='10X', color='#52bcec', marker='^', alpha=0.4)
plt.scatter(tissue_percentage, iou1, label='Vispro', color='#501d8a', marker='o', s=40, alpha=0.4)
plt.scatter(tissue_percentage, iou2, label='Baseline U-Net', color='#1c8041', marker='o', alpha=0.4)
plt.scatter(tissue_percentage, iou3, label='10X', color='#e55709', marker='o', alpha=0.4)
# Plot averaged IoU curves with larger intervals
# plt.plot(x_positions, [avg_iou1[label] for label in range_labels], color='#e6b532', marker='o', linestyle='-', linewidth=2)
# plt.plot(x_positions, [avg_iou2[label] for label in range_labels], color='#d75425', marker='s', linestyle='-', linewidth=2)
# plt.plot(x_positions, [avg_iou3[label] for label in range_labels], color='#52bcec', marker='^', linestyle='-', linewidth=2)






for x, label in zip(x_positions, range_labels):
    y1 = avg_iou1[label]
    y2 = avg_iou2[label]
    y3 = avg_iou3[label]
    plt.plot([x, x], [y1, y2], color='#501d8a', linestyle='--', linewidth=1)  # Dashed line
    # plt.text(x, y, f"{label}\n{y:.2f}", ha='center', va='bottom', fontsize=10, color='#501d8a')  # Label
    plt.plot([x, x], [y2, y3], color='#1c8041', linestyle='--', linewidth=1)  # Dashed line
    # plt.plot([x, x], [y3, 0], color='#e55709', linestyle='--', linewidth=1)  # Dashed line

plt.plot(x_positions, [avg_iou1[label] for label in range_labels], color='#501d8a', marker='^',markersize=8, linestyle='-', linewidth=2)
plt.plot(x_positions, [avg_iou2[label] for label in range_labels], color='#1c8041', marker='s',markersize=8, linestyle='-', linewidth=2)
plt.plot(x_positions, [avg_iou3[label] for label in range_labels], color='#e55709', marker='x',markersize=8,linestyle='-', linewidth=2)



#

# Customize x-axis and y-axis
ax = plt.gca()
ax.set_xscale('custom_x')

# ax.set_yscale('custom_y')

# Add background color for each interval
# colors = ['#c6d182','#eae0e9', '#e0c7e3', '#ae98b6', '#846e89']
# ax.axvspan(-0.05, 0.05, facecolor=colors[0], alpha=0.4, label='No overlapping')
# ax.axvspan(0.05, 10, facecolor=colors[1], alpha=0.4, label='Slight overlapping')
# ax.axvspan(10, 30, facecolor=colors[2], alpha=0.4, label='Moderate overlapping')
# ax.axvspan(30, 50, facecolor=colors[3], alpha=0.4, label='High overlapping')
# ax.axvspan(50, 100, facecolor=colors[4], alpha=0.4, label='Very high overlapping')
# ax.text(-0.003, 1.005, "No overlap", fontsize=14, color='black')
# ax.text(2.7, 1.005, "Slight overlap", fontsize=14, color='black')
# ax.text(12.04, 1.005, "Moderate overlap", fontsize=14, color='black')
# ax.text(34.04, 1.005, "High overlap", fontsize=14, color='black')
# ax.text(55.04, 1.005, "Very high overlap", fontsize=14, color='black')


# Set custom x and y ticks with bold font
# ax.set_xscale('symlog')
ax.set_xlim(-0.01, 105)
ax.set_xticks([0, 10, 30, 50, 100])
# ax.set_xticklabels(["0%","10%", "30%", "50%", "100%"],
#                    fontsize=16)
ax.set_xticklabels(["0% ", "10%", "30% ", "50%", "100%"],
                   fontsize=12)


ax.set_yticks([0.2,0.3,0.4, 0.5,0.6,0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(['0.2','0.3','0.4', '0.5','0.6','0.7', '0.8', '0.9', '1.0'], fontsize=16)

ax.set_ylim(0.24, 1)
# Set axis labels and title with bold font
plt.xlabel('Marker Overlapping Percentage with Tissue Area', fontsize=16)
plt.ylabel('IoU', fontsize=16)

# Configure the legend with bold font
ax.legend(loc="lower right", fontsize=12, frameon=True,bbox_to_anchor=(0.98, -0.01))

# Set global font properties for all text in the figure
plt.rcParams.update({'font.size': 16})


plt.savefig('./figures/3.png', dpi=300)
plt.show()
