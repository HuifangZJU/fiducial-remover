import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1) Read your Excel data ---
df = pd.read_excel("/home/huifang/workspace/paper/Supplementary Table S2.xlsx")
df.columns = [c.strip() for c in df.columns]  # strip trailing spaces

# Suppose we only need these columns
df = df[["Species", "Tissue type", "Conditions", "Number of images"]]

# --- 2) Categorize the "Conditions" ---
def condition_category(cond_str):
    c_lower = str(cond_str).lower()
    if any(x in c_lower for x in ["cancer", "carcinoma", "tumor", "lymphoma"]):
        return "Cancer"
    elif "normal" in c_lower or "control" in c_lower:
        return "Normal"
    elif any(x in c_lower for x in ["injury", "disease", "colitis", "vasculitis",
                                    "schizophrenia", "autism", "hyperplasia",
                                    "lepromatous", "resection"]):
        return "Disease/Pathology"
    elif any(x in c_lower for x in ["menstrual", "postnatal", "budding",
                                    "development", "stage"]):
        return "Development/Physiology"
    else:
        return "Other"

df["Condition Category"] = df["Conditions"].apply(condition_category)

# --- 3) Summarize categories and sort ascending (smallest->largest) ---
cat_summary = df.groupby("Condition Category")["Number of images"].sum()
cat_summary = cat_summary.sort_values(ascending=True)

labels = cat_summary.index.tolist()   # Category names
counts = cat_summary.values.tolist()  # Corresponding counts

# --- 4) Convert counts to percentages ---
total_count = sum(counts)
percentages = [c / total_count * 100 for c in counts]

# --- 5) Create angles for each category, from 0 to 2π (one segment per category) ---
angles = np.linspace(0, 2 * np.pi, len(counts) + 1, endpoint=True)

# --- 6) Create subtle differences in bar lengths (optional, to mimic your style) ---
adjusted_counts = [x * 0.1 + 10 for x in counts]

# --- 7) Set up the polar plot ---
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': 'polar'})

# We’ll define a color palette (repeat if there are more categories than colors)
base_colors = ['#c6d182', '#eae0e9', '#e0c7e3', '#ae98b6', '#846e89']
if len(counts) > len(base_colors):
    base_colors = (base_colors * (len(counts) // len(base_colors) + 1))[:len(counts)]

# Plot bars
bars = ax.bar(
    angles[:-1],            # Starting angle of each bar
    adjusted_counts,         # The radial "length" of each bar
    width = angles[1] - angles[0],
    color = base_colors,
    edgecolor='white',
    linewidth=1.5,
    alpha=0.85
)

# --- 8) Annotate each segment with category label & percentage ---
for angle, radius, label, pct in zip(angles[:-1], adjusted_counts, labels, percentages):
    # Place the category label near the outer edge of each bar
    ax.text(
        angle,
        radius + 3,
        label,
        ha='center',
        va='center',
        fontsize=24,
        color='black'
    )
    # Place the percentage roughly in the middle of the bar
    ax.text(
        angle,
        radius / 2,
        f"{pct:.1f}%",
        ha='center',
        va='center',
        fontsize=26,
        color='black',
        fontweight='bold'
    )

# --- 9) Style adjustments ---
ax.set_yticklabels([])  # Hide radial (distance) labels
ax.set_xticks([])       # Hide angle tick labels
ax.spines['polar'].set_visible(False)  # Hide the circular outline
ax.grid(False)          # Turn off radial grid

# **Ensure the smallest category is at the top, going clockwise**:
ax.set_theta_zero_location("N")  # 0 degrees at "North"/top
ax.set_theta_direction(-1)       # proceed clockwise

plt.title("Condition Distribution (Smallest Category at Top, Clockwise)",
          fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig('./figures/2.png', dpi=300)
plt.show()
