import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your CSV file
df = pd.read_csv("nandpuf_50runscsv.csv")  # Change filename if needed

# Filter for rows with iter0 only (1st round of each bit)
df_iter0 = df[df["Output"].str.contains(r"_iter0$", na=False)].copy()

# Convert scientific notation strings like "800e-3" to floats
df_iter0["Nominal"] = pd.to_numeric(df_iter0["Nominal"].astype(str).str.replace("e", "E"), errors="coerce")

# Drop any rows where conversion failed
values = df_iter0["Nominal"].dropna()

# Set seaborn style
sns.set(style="whitegrid")

# Create the figure
fig, ax1 = plt.subplots(figsize=(7, 4))

# Plot histogram on primary axis
sns.histplot(values, bins=50, kde=False, color="mediumpurple", edgecolor="black", ax=ax1)
ax1.set_xlabel("Nominal Value (V)")
ax1.set_ylabel("Frequency", color="black")
ax1.tick_params(axis='y', labelcolor='black')

# Plot KDE on secondary axis
ax2 = ax1.twinx()
sns.kdeplot(values, ax=ax2, color="black", linewidth=2, label="KDE")
ax2.set_ylabel("Density", color="black")
ax2.tick_params(axis='y', labelcolor='black')

# Add statistics annotation
mu = values.mean()
std = values.std()
ax1.text(0.95, 0.95, f"μ = {mu:.3g}\nσ = {std:.3g}\nN = {len(values)}",
         transform=ax1.transAxes, ha='right', va='top',
         fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

# Final touches
ax1.set_title("Histogram + KDE of Nominal Values at iter0 (All 128 Bits)")
ax2.legend(loc="upper left")
fig.tight_layout()
plt.show()
