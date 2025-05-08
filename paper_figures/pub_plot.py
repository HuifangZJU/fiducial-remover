import pandas as pd
import matplotlib.pyplot as plt

# 1) Define your tools and years
tools = [
    "Space Ranger","Loupe Browser","SpaCell","stLearn","SpaGCN",
    "TIST","Hist2ST","GraphST","ConGcR",
    "SiGRA","stMMC","iIMPACT","GIST","STAIG"
]
years = [2019,2019,2020,2021,2021,2022,2022,2023,2023,2023,2024,2024,2024,2025]

# 2) Define your image‑usage groups
usage = [
    "Fiducial alignment & masking", "Cluster overlay for QC",
    "CNN features + gene concat", "VGG16 morphological distances",
    "RGB/CNN spot attributes", "Haralick texture stats",
    "Transformer tile encoding", "ResNet tile embeddings",
    "Contrastive image encoder", "ResNet‑18 cell‑level features",
    "ResNet‑34 tile features", "PCA on H&E patches",
    "CLIP‑style patch embeddings", "Self‑sup ResNet features"
]

# 3) For each tool, fill in whether it has each feature.
#    Replace True/False below with your own assessments.
user_friendly = [ True,  True,  False, True,  False,
                  False, True,  False, False, True,
                  True,  True,  True,  True ]
integrates_image = [ False, False, True,  True,  True,
                     True,  True,  True,  True,  True,
                     True,  True,  True,  True ]
tissue_area_id = [ True,  False, False, True,  True,
                   True,  False, False, False, True,
                   True,  False, False, True ]
easy_output = [ True,  True,  False, True,  False,
                False, True,  False, False, True,
                True,  True,  True,  True ]

# 4) Build DataFrame
df = pd.DataFrame({
    "Tool": tools,
    "Year": years,
    "Image Usage Group": usage,
    "Batch processing": user_friendly,
    "Integrates image module": integrates_image,
    "Tissue-area identification": tissue_area_id,
    "Easy downstream output": easy_output
})

# Convert booleans to ticks
for col in ["Batch processing","Integrates image module",
            "Tissue-area identification","Easy downstream output"]:
    df[col] = df[col].map({True:"✓", False:"✗"})

# 5) Plot as a styled table
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')

table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc='center',
    cellLoc='center'
)
# automatically size every column to its contents
table.auto_set_column_width(col=list(range(len(df.columns))))

# 6) Style with your base colors
base_colors = ['#e0c7e3','#eae0e9','#c6d182','#ae98b6','#846e89']
header_bg   = base_colors[2]  # green
row_bg      = [base_colors[1], 'white']

table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 1.5)

# Color header & rows, and highlight ✓/✗
for (r,c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor(header_bg)
        cell.set_text_props(weight='bold', color='white')
    else:
        cell.set_facecolor(row_bg[r%2])
        # make checks green, crosses red
        if c >= df.columns.get_loc("Batch processing"):
            txt = cell.get_text().get_text()
            if txt == "✓":
                cell.get_text().set_color('green')
                cell.get_text().set_weight('bold')
            else:
                cell.get_text().set_color('red')
                cell.get_text().set_weight('bold')

# 7) Title
fig.suptitle(
    "ST Tools: Image‑Usage Categories & Key Features",
    fontsize=14, fontweight='bold'
)
plt.tight_layout(rect=[0,0,1,0.93])
plt.show()
