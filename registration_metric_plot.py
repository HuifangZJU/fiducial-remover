import matplotlib.pyplot as plt
import numpy as np

# Sample data: replace these with your actual results
metrics_algorithm1 = {'SSIM': 0.63, 'Mutual Information': 0.60}
metrics_algorithm2 = {'SSIM': 0.66, 'Mutual Information': 0.68}
# metrics_algorithm1 = {'mean IOU': 0.85, 'mean DSC': 0.90}
# metrics_algorithm2 = {'mean IOU': 0.94, 'mean DSC': 0.97}

# Names of the metrics
metric_names = list(metrics_algorithm1.keys())

# Values for each algorithm
values_algorithm1 = list(metrics_algorithm1.values())
values_algorithm2 = list(metrics_algorithm2.values())

# Set up the bar graph
x = np.arange(len(metric_names))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, values_algorithm1, width, label='with fiducial')
bars2 = ax.bar(x + width/2, values_algorithm2, width, label='without fiducial')

# Add some text for labels, title, and custom x-axis tick labels
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
# ax.set_title('Comparison of Image Registration Algorithms')
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.legend()

# # Set the y-axis to only show from 0.5 to 1.0
ax.set_ylim(0.3, 0.8)

# Function to add a label on top of each bar
def autolabel(bars):
    """Attach a text label above each bar displaying its height."""
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Call the function to add labels
autolabel(bars1)
autolabel(bars2)

# Show the plot
plt.show()