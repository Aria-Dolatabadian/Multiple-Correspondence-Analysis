
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prince import MCA
import matplotlib.patches as mpatches

aggressiveness_levels = ["Aggressive", "Moderately Aggressive", "Hypo Aggressive"]
six_genes = [f"SIX{i}" for i in range(1, 13)]  # Exactly 12 SIX genes

# Read Data from CSV
df = pd.read_csv("aggressiveness_six_genes.csv")

# Encode the categorical variable (Aggressiveness) for MCA
df["Aggressiveness"] = df["Aggressiveness"].astype("category")

#Perform MCA
mca = MCA(n_components=2)
mca_results = mca.fit(df)

# Get row and column coordinates
row_coords = mca.transform(df)
col_coords = mca.column_coordinates(df)

# **Filter SIX genes to avoid multiple categories (SIX1_0, SIX1_1, etc.)**
filtered_six_genes = [col for col in col_coords.index if any(gene in col for gene in six_genes)]
col_coords = col_coords.loc[filtered_six_genes]

# **Keep unique SIX genes for the legend**
unique_six_genes = sorted(set([col.split("_")[0] for col in col_coords.index]))  # Extract unique SIX names

#Visualize MCA Results as a Biplot
fig, ax = plt.subplots(figsize=(8, 8))

# Define unique colors for each SIX gene
colors = sns.color_palette("husl", len(unique_six_genes))

# **Show Aggressiveness levels in the plot**
scatter = sns.scatterplot(x=row_coords[0], y=row_coords[1], hue=df["Aggressiveness"], palette="coolwarm", alpha=0.6, edgecolor="black")

# Plot variable categories as arrows (SIX genes)
for i, col in enumerate(col_coords.index):
    gene_name = col.split("_")[0]  # Keep only "SIX1", "SIX2", etc.
    if gene_name in unique_six_genes:  # Ensure each gene is plotted only once
        plt.arrow(0, 0, col_coords.loc[col, 0], col_coords.loc[col, 1],
                  color=colors[i % len(colors)], alpha=0.7, head_width=0.02)
        plt.text(col_coords.loc[col, 0] * 1.1, col_coords.loc[col, 1] * 1.1,
                 gene_name, color=colors[i % len(colors)], fontsize=12)

# **Create legends for easy positioning**
gene_patches = [mpatches.Patch(color=colors[i], label=gene) for i, gene in enumerate(unique_six_genes)]
handles_aggr, labels_aggr = scatter.get_legend_handles_labels()

# **Legend positions (Customizable using (x, y) coordinates)**
legend_aggr_pos = (1.05, 0.8)  # Move Aggressiveness legend
legend_six_pos = (1.05, 0.4)   # Move SIX genes legend

# **Legend for Aggressiveness**
legend_aggr = plt.legend(handles=handles_aggr, labels=labels_aggr, title="Aggressiveness",
                         bbox_to_anchor=legend_aggr_pos, loc="center left", borderaxespad=0)
plt.gca().add_artist(legend_aggr)

# **Legend for SIX Genes**
legend_six = plt.legend(handles=gene_patches, title="SIX Genes",
                        bbox_to_anchor=legend_six_pos, loc="center left", borderaxespad=0)

# Add a circle (like PCA)
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='dashed')
ax.add_patch(circle)

plt.axhline(0, color='black', linewidth=0.5, linestyle="--")
plt.axvline(0, color='black', linewidth=0.5, linestyle="--")
plt.xlabel("MCA Dimension 1")
plt.ylabel("MCA Dimension 2")
plt.title("MCA Biplot: Aggressiveness vs SIX Genes")

plt.grid(True)
plt.show()
