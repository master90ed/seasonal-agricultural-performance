import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load & Clean
df = pd.read_excel("seasonal_agriculture_performance_dataset.xlsx")
for col in ["Rainfall_mm", "Soil_Moisture_pct", "Yield_Tonnes_Ha"]:
    df[col].fillna(df[col].median(), inplace=True)

season_order = ["Kharif", "Rabi", "Zaid"]
df["Season"] = pd.Categorical(df["Season"], categories=season_order, ordered=True)

print(df.describe())
print(df["Season"].value_counts())

# Seasonal Summary
summary = df.groupby("Season", observed=True)[
    ["Yield_Tonnes_Ha", "Profit_INR", "Rainfall_mm", "Disease_Pest_Risk_pct"]
].mean().round(2)
print(summary)

# Bar Charts – Seasonal Overview
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.flat, ["Yield_Tonnes_Ha", "Profit_INR", "Rainfall_mm",
                                "Revenue_INR", "Disease_Pest_Risk_pct", "Water_Efficiency_t_per_1000m3"]):
    df.groupby("Season", observed=True)[col].mean().plot(kind="bar", ax=ax, edgecolor="white")
    ax.set_title(col); ax.set_xlabel(""); plt.setp(ax.get_xticklabels(), rotation=0)
plt.suptitle("Seasonal Overview", fontweight="bold"); plt.tight_layout(); plt.show()

# Yield Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="Season", y="Yield_Tonnes_Ha", palette="Set2", ax=ax1)
sns.violinplot(data=df, x="Season", y="Yield_Tonnes_Ha", palette="Set2", inner="quartile", ax=ax2)
ax1.set_title("Yield Box Plot"); ax2.set_title("Yield Violin Plot")
plt.tight_layout(); plt.show()

# Crop Yield by Season
df.groupby(["Season", "Crop"], observed=True)["Yield_Tonnes_Ha"].mean().unstack().T.plot(
    kind="bar", figsize=(14, 5), colormap="tab10", edgecolor="white")
plt.title("Avg Yield by Crop & Season"); plt.xticks(rotation=45, ha="right")
plt.legend(title="Season"); plt.tight_layout(); plt.show()

# Correlation Heatmap
num_cols = ["Rainfall_mm", "Avg_Temperature_C", "Humidity_pct", "Soil_pH",
            "Fertilizer_kg_ha", "Seed_Quality_Score", "Yield_Tonnes_Ha",
            "Profit_INR", "Disease_Pest_Risk_pct"]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Heatmap"); plt.tight_layout(); plt.show()

# Scatter – Yield vs Key Factors
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, col in zip(axes.flat, ["Rainfall_mm", "Seed_Quality_Score", "Fertilizer_kg_ha", "Soil_Moisture_pct"]):
    sns.scatterplot(data=df, x=col, y="Yield_Tonnes_Ha", hue="Season", alpha=0.4, s=15, ax=ax)
    ax.set_title(f"Yield vs {col}")
plt.tight_layout(); plt.show()

# Irrigation vs Yield & Profit
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col in zip(axes, ["Yield_Tonnes_Ha", "Profit_INR"]):
    df.groupby(["Season", "Irrigation_Method"], observed=True)[col].mean().unstack().plot(
        kind="bar", ax=ax, edgecolor="white")
    ax.set_title(col); plt.setp(ax.get_xticklabels(), rotation=0)
plt.suptitle("Irrigation Impact", fontweight="bold"); plt.tight_layout(); plt.show()

# Profitability
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
df.groupby("Season", observed=True)["Profit_INR"].apply(lambda x: (x > 0).mean() * 100).plot(
    kind="bar", ax=ax1, edgecolor="white")
ax1.set_title("% Profitable Farms"); ax1.set_ylim(0, 100)
sns.boxplot(data=df, x="Season", y="Profit_INR", palette="Set2", ax=ax2)
ax2.axhline(0, color="red", linestyle="--", label="Break-even"); ax2.legend()
ax2.set_title("Profit Distribution"); plt.tight_layout(); plt.show()

# Outlier Detection
Q1, Q3 = df["Yield_Tonnes_Ha"].quantile(0.25), df["Yield_Tonnes_Ha"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["Yield_Tonnes_Ha"] < Q1 - 1.5*IQR) | (df["Yield_Tonnes_Ha"] > Q3 + 1.5*IQR)]
print(f"Outliers: {len(outliers)} farms")
fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(df.index, df["Yield_Tonnes_Ha"], s=10, alpha=0.3, label="Normal")
ax.scatter(outliers.index, outliers["Yield_Tonnes_Ha"], color="red", s=20, label="Outlier")
ax.set_title("Yield Outliers (IQR)"); ax.legend(); plt.tight_layout(); plt.show()

# Disease Risk
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df, x="Season", y="Disease_Pest_Risk_pct", palette="Reds", ax=ax1)
df.groupby("Crop")["Disease_Pest_Risk_pct"].mean().sort_values().plot(kind="barh", ax=ax2, color="salmon")
ax1.set_title("Disease Risk by Season"); ax2.set_title("Disease Risk by Crop")
plt.tight_layout(); plt.show()

# ANOVA – F-statistic
groups = [g["Yield_Tonnes_Ha"].values for _, g in df.groupby("Season", observed=True)]
gm = df["Yield_Tonnes_Ha"].mean(); k, N = len(groups), len(df)
F = (sum(len(g)*(g.mean()-gm)**2 for g in groups)/(k-1)) / (sum(((g-g.mean())**2).sum() for g in groups)/(N-k))
print(f"ANOVA F = {F:.4f} → {'Significant' if F > 3 else 'Not significant'} yield difference across seasons")

# Summary
print(f"\nBest yield season : {summary['Yield_Tonnes_Ha'].idxmax()}")
print(f"Best profit season: {summary['Profit_INR'].idxmax()}")
print(f"Best crop         : {df.groupby('Crop')['Yield_Tonnes_Ha'].mean().idxmax()}")
print(f"Best irrigation   : {df.groupby('Irrigation_Method')['Yield_Tonnes_Ha'].mean().idxmax()}")
print(f"High risk season  : {summary['Disease_Pest_Risk_pct'].idxmax()}")
