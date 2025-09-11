import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
df = pd.read_csv("ExplorerRun.0.csv")  # Replace with actual filename
df["Nominal"] = pd.to_numeric(df["Nominal"], errors="coerce")
values = df["Nominal"].dropna()

# Set seaborn style
sns.set(style="whitegrid")

# Create figure
fig, ax1 = plt.subplots(figsize=(6, 4))

# Plot histogram (frequency)
sns.histplot(values, bins=50, kde=False, color="mediumpurple", edgecolor="black", ax=ax1)
ax1.set_xlabel("Nominal Value")
ax1.set_ylabel("Frequency", color="black")
ax1.tick_params(axis='y', labelcolor='black')

# Create second y-axis for KDE (density)
ax2 = ax1.twinx()
sns.kdeplot(values, ax=ax2, color="black", linewidth=2, label="KDE")
ax2.set_ylabel("Density", color="black")
ax2.tick_params(axis='y', labelcolor='black')

# Stats annotation (optional)
mu = values.mean()
std = values.std()
ax1.text(0.95, 0.95, f"μ = {mu:.3g}\nσ = {std:.3g}\nN = {len(values)}",
         transform=ax1.transAxes, ha='right', va='top',
         fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

# Title, legend, layout
ax1.set_title("Histogram with KDE of Nominal Output Values")
ax2.legend(loc="upper left")
fig.tight_layout()
plt.show()
