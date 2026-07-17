import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

train_df = pd.read_csv("dataset_kaggle/exoTrain.csv")

# Same exoplanet star as before
exoplanet_row = train_df[train_df["LABEL"] == 2].iloc[0]
flux = exoplanet_row.drop("LABEL").values.astype(float)

# --- Detrend using Savitzky-Golay filter ---
# window_length must be odd; 301 gives a smooth long-term trend estimate
trend = savgol_filter(flux, window_length=301, polyorder=2)
flattened = flux - trend

# --- Normalize (z-score: mean 0, std 1) ---
normalized = (flattened - np.mean(flattened)) / np.std(flattened)

# Plot: raw vs trend vs flattened+normalized
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(flux, color="blue")
axes[0].set_title("Raw Flux")

axes[1].plot(flux, color="blue", alpha=0.5, label="raw")
axes[1].plot(trend, color="red", label="detected trend")
axes[1].set_title("Raw Flux with Detected Trend Overlaid")
axes[1].legend()

axes[2].plot(normalized, color="green")
axes[2].set_title("Flattened + Normalized (trend removed)")

plt.tight_layout()
plt.show()