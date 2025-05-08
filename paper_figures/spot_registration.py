import matplotlib.pyplot as plt
import numpy as np
# palette = {
#     'Original': '#cccccc',
#     'PASTE': '#8da0cb',
#     'Vispro': '#fc8d62'
# }

# # Data
# layers = ['Layer3', 'Layer4', 'Layer5', 'Layer6', 'WM']
# original = [0.78, 0.00, 0.39, 0.58, 0.60]
# paste =    [0.89, 0.51, 0.78, 0.87, 0.89]
# vispro =   [0.90, 0.61, 0.86, 0.88, 0.89]
#
# # X locations
# x = np.arange(len(layers))
# bar_width = 0.25
# base = ['#c6d182','#e0c7e3','#eae0e9','#ae98b6','#846e89']
# # Custom colors (replace with your preferred palette if available)
# palette = {
#     'Original': base[3],
#     'PASTE': base[4],
#     'Vispro': base[0]
# }
#
# # Create plot
#
# plt.rcParams.update({'font.size': 16})
# fig, ax = plt.subplots(figsize=(7, 5))
# ax.bar(x - bar_width, original, width=bar_width, label='Original', color=palette['Original'])
# ax.bar(x,             paste,    width=bar_width, label='PASTE',    color=palette['PASTE'])
# ax.bar(x + bar_width, vispro,   width=bar_width, label='Vispro',   color=palette['Vispro'])
#
# # Axes & Labels
# ax.set_xticks(x)
# ax.set_xticklabels(layers,fontsize=14)
# ax.set_ylabel('Dice Score')
# ax.set_ylim(0, 1.0)
# ax.set_title('Pair AB')
# ax.legend(frameon=False,fontsize=14,loc='center left', bbox_to_anchor=(1, 0.5))
# ax.grid(axis='y', linestyle='--', alpha=0.4)
# plt.tight_layout()
# plt.savefig('./figures/14.png', dpi=300)
# plt.show()


# Data
layers = ['Layer3', 'Layer4', 'Layer5', 'Layer6', 'WM']
original = [0.93, 0.44, 0.71, 0.63, 0.82]
paste =    [0.93, 0.67, 0.83, 0.85, 0.93]
vispro =   [0.95, 0.64, 0.81, 0.79, 0.90]

# X locations
x = np.arange(len(layers))
bar_width = 0.25
base = ['#c6d182','#e0c7e3','#eae0e9','#ae98b6','#846e89']
# Custom colors (replace with your preferred palette if available)
palette = {
    'Original': base[3],
    'PASTE': base[4],
    'Vispro': base[0]
}

# Create plot

plt.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - bar_width, original, width=bar_width, label='Original', color=palette['Original'])
ax.bar(x,             paste,    width=bar_width, label='PASTE',    color=palette['PASTE'])
ax.bar(x + bar_width, vispro,   width=bar_width, label='Vispro',   color=palette['Vispro'])

# Axes & Labels
ax.set_xticks(x)
ax.set_xticklabels(layers,fontsize=14)
ax.set_ylabel('Dice Score')
ax.set_ylim(0, 1.0)
ax.set_title('Pair CD')
ax.legend(frameon=False,fontsize=14,loc='center left', bbox_to_anchor=(1, 0.5))
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
# plt.savefig('./figures/15.png', dpi=300)
plt.show()