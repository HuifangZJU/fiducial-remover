import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

###########################
# 1) Load & Prep Data
###########################
df = pd.read_excel("/home/huifang/workspace/paper/Supplementary Table S2.xlsx")
df.columns = [c.strip() for c in df.columns]

df = df[["Species", "Tissue type", "Conditions", "Number of images"]]

def midpoint_angle_deg(wedge):
    return 0.5 * (wedge.theta1 + wedge.theta2)

# Map species into 3 categories (Homo sapiens, Mus musculus, Other)
def map_species(sp):
    sp_lower = str(sp).lower()
    if "homo" in sp_lower:
        return "Homo sapiens"
    elif "mus" in sp_lower:
        return "Mus musculus"
    else:
        return "Other"


df["Species"] = df["Species"].apply(map_species)

# Species totals (inner ring)
species_totals = df.groupby("Species")["Number of images"].sum().sort_values(ascending=False)
species_labels = species_totals.index.tolist()
species_sizes = species_totals.values.tolist()
total_images = sum(species_sizes)

###########################
# 2) Outer Ring: Tissue
###########################
species_tissue = df.groupby(["Species", "Tissue type"])["Number of images"].sum()

# We'll store outer slices in a list: (species, fraction, label_text, color, needs_line)
outer_slices = []

# Define color palette
base_colors = ['#e0c7e3','#eae0e9', '#c6d182',  '#ae98b6', '#846e89']

def hex_to_rgb(hexcol):
    hexcol = hexcol.lstrip('#')
    return tuple(int(hexcol[i:i+2], 16) / 255 for i in (0, 2, 4))

# Assign each species a base color
species_color_map = {}
for i, sp in enumerate(species_labels):
    species_color_map[sp] = base_colors[i % len(base_colors)]

# Numeric threshold for leader lines
threshold_count = 5

for sp in species_labels:
    # Base color for this species
    sp_base = species_color_map[sp]
    br, bg, bb = hex_to_rgb(sp_base)

    # Tissue dictionary for this species => { tissue_name: count }
    if sp in species_tissue.index.levels[0]:
        sub = species_tissue.loc[sp].to_dict()
    else:
        sub = {}

    # 1) Group all tissues with exactly 1 image into a single "Others" entry
    others_sum = 0
    tissues_to_remove = []
    for tissue_name, tcount in sub.items():
        if tcount == 1:
            others_sum += 1
            tissues_to_remove.append(tissue_name)

    # Remove them from sub
    for tname in tissues_to_remove:
        del sub[tname]

    # If we found any with count==1, add them together as "Others"
    if others_sum > 0:
        sub["Others"] = sub.get("Others", 0) + others_sum

    # 2) Sort tissues by descending count
    sorted_tissues = sorted(sub.items(), key=lambda x: x[1], reverse=True)

    # 3) Build final outer_slices list
    for tissue_name, tcount in sorted_tissues:
        fraction = tcount / total_images
        label_str = f"{tissue_name} ({tcount})"

        # Tint from species color
        alpha_factor = 0.7
        color_rgba = (br * alpha_factor, bg * alpha_factor, bb * alpha_factor, 1.0)

        # If tcount < threshold_count => use leader line
        needs_line = (tcount < threshold_count)

        outer_slices.append((sp, fraction, label_str, color_rgba, needs_line))

# Convert to lists for plotting
outer_fracs = [o[1] for o in outer_slices]
outer_colors = [o[3] for o in outer_slices]

###########################
# 3) Plot the Nested Pie
###########################
fig, ax = plt.subplots(figsize=(15, 8))

# === INNER RING (Species) ===
# Donut: from radius=0.4 to 0.7 (width=0.3)
wedges_inner, _ = ax.pie(
    species_sizes,
    labels=None,  # <-- Disable built-in text labels
    colors=[species_color_map[s] for s in species_labels],
    startangle=0,
    radius=0.5,  # Outer radius of the inner ring
    wedgeprops=dict(width=0.25, edgecolor='white')
)

# The inner ring then extends from radius=0.4 to 0.7 (a width of 0.3).
# The midpoint is about (0.4 + 0.7)/2 = 0.55 if you want to place text exactly in the center.

ring_mid_radius = 0.28
for wedge, label in zip(wedges_inner, species_labels):
    # 1) Find the wedge’s midpoint angle
    angle_deg = midpoint_angle_deg(wedge)
    angle_rad = np.deg2rad(angle_deg)

    # 2) Compute x,y at your desired radius (here ring_mid_radius=0.55)
    x_label = ring_mid_radius * np.cos(angle_rad)
    y_label = ring_mid_radius * np.sin(angle_rad)

    # 3) Place text with any styling you want
    ax.text(
        x_label,
        y_label,
        label,
        ha='center',
        va='center',
        fontsize=14,
        fontweight='bold'
    )
# === OUTER RING (Tissues) ===
# Donut: from radius=0.7 to 1.0 (width=0.3)
wedges_outer, _ = ax.pie(
    outer_fracs,
    labels=None,
    colors=outer_colors,
    startangle=0,
    radius=0.8,
    wedgeprops=dict(width=0.35, edgecolor='white')
)

ax.set_title("Nested Pie: Species (inner) vs Tissue (outer)", fontsize=14)
ax.set_aspect("equal")


###########################
# 3) Draw the 2-Segment Leader Lines
###########################



for wedge, (sp, fraction, label_str, color_rgba, needs_line) in zip(wedges_outer, outer_slices):
    angle_deg = midpoint_angle_deg(wedge)
    angle_rad = np.deg2rad(angle_deg)

    # Middle radius of outer donut is ~0.85
    r_mid = 0.65
    x_wedge = r_mid * np.cos(angle_rad)
    y_wedge = r_mid * np.sin(angle_rad)

    if needs_line:
        # We'll draw 2 line segments:
        #  1) wedge center -> "mid" radial point
        #  2) mid point -> text location (horizontal)

        # 1) Let's push the "midpoint" further out radially by ~10%:
        scale_factor = 1.2
        x_mid = x_wedge * scale_factor
        y_mid = y_wedge * scale_factor

        # 2) Decide left or right
        if angle_deg > 270 or angle_deg<90:
            x_text = 0.9
            ha = 'left'
        else:
            x_text = -0.9
            ha = 'right'
        y_text = y_mid  # keep the same Y to produce a horizontal line

        # Draw the 1st segment (radial extension)
        ax.plot([x_wedge, x_mid], [y_wedge, y_mid], color='gray')
        # Draw the 2nd segment (horizontal)
        ax.plot([x_mid, x_text], [y_mid, y_text], color='gray')

        # Place the text at (x_text, y_text)
        ax.text(x_text, y_text, label_str, ha=ha, va='center', fontsize=11)
    else:
        # For bigger slices, place text inside the wedge
        ax.text(x_wedge, y_wedge, label_str, ha='center', va='center', fontsize=11)

plt.tight_layout()
plt.savefig('./figures/3.png', dpi=300)
plt.show()
