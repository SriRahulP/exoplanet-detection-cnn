import pandas as pd
import matplotlib.pyplot as plt

# Load training data
train_df = pd.read_csv("dataset_kaggle/exoTrain.csv")

# Grab one row that IS an exoplanet (LABEL == 2)
exoplanet_row = train_df[train_df["LABEL"] == 2].iloc[0]

# Grab one row that is NOT an exoplanet (LABEL == 1)
non_exoplanet_row = train_df[train_df["LABEL"] == 1].iloc[0]

# The flux values are every column except LABEL
exoplanet_flux = exoplanet_row.drop("LABEL").values
non_exoplanet_flux = non_exoplanet_row.drop("LABEL").values

# Plot both side by side
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(exoplanet_flux, color="blue")
axes[0].set_title("Light Curve - CONFIRMED EXOPLANET STAR")
axes[0].set_xlabel("Time step")
axes[0].set_ylabel("Flux (brightness)")

axes[1].plot(non_exoplanet_flux, color="gray")
axes[1].set_title("Light Curve - NON-EXOPLANET STAR")
axes[1].set_xlabel("Time step")
axes[1].set_ylabel("Flux (brightness)")

plt.tight_layout()
plt.show()