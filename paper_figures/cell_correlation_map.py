import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── your data ───────────────────────────────────────────
corr_vispro = np.array([
    -0.141, 0.345, -0.132, 0.326, 0.246, -0.049, 0.280, 0.032, 0.021,
     0.011, 0.007, -0.014, 0.009, -0.034, -0.035, -0.008, -0.008, 0.011, -0.017
])
corr_original = np.array([
    -0.140, 0.351, -0.156, 0.339, 0.217, -0.050, 0.215, 0.023, 0.015,
     0.003, -0.010, -0.011, 0.008, -0.019, -0.044, -0.009, -0.002, 0.003, -0.024
])

data   = [corr_vispro, corr_original]
labels = ['Vispro', 'Original']
colors = ['#c6d182', '#ae98b6']            # greenish & muted purple

# ── figure layout ───────────────────────────────────────
fig = plt.figure(figsize=(7, 6))
gs  = fig.add_gridspec(2, 1, height_ratios=[3, 1])

ax_main  = fig.add_subplot(gs[0])
ax_inset = fig.add_subplot(gs[1])

# ── main panel: violin + box overlay ────────────────────
sns.violinplot(data=data,
               palette=colors,
               cut=0, inner=None,
               ax=ax_main)
sns.boxplot(data=data,
            palette=colors,
            width=0.15,
            showcaps=True,
            boxprops={'zorder':3},
            ax=ax_main)

ax_main.set_xticklabels(labels, fontsize=11)
ax_main.set_ylabel('Correlation', fontsize=12)
ax_main.set_title('Distribution of spot‑wise correlation', fontsize=13)
ax_main.grid(axis='y', linestyle='--', alpha=0.4)

# ── inset: overlayed histogram + KDE ─────────────────────
bins = np.linspace(min(corr_vispro.min(), corr_original.min()),
                   max(corr_vispro.max(), corr_original.max()), 12)

for arr, col, lab in zip(data, colors, labels):
    ax_inset.hist(arr, bins=bins, color=col, alpha=0.4, label=lab, density=True)
    sns.kdeplot(arr, color=col, ax=ax_inset)

ax_inset.set_xlabel('Correlation')
ax_inset.set_ylabel('Density')
ax_inset.grid(axis='y', linestyle='--', alpha=0.4)
ax_inset.legend(frameon=False)

plt.tight_layout()
plt.show()
