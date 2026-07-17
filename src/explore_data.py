import pandas as pd

# Load the datasets
train_df = pd.read_csv("dataset_kaggle/exoTrain.csv")
test_df = pd.read_csv("dataset_kaggle/exoTest.csv")

# Basic shape check
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# The first column is the LABEL (2 = confirmed exoplanet, 1 = not an exoplanet)
print("\nTrain label counts:")
print(train_df["LABEL"].value_counts())

print("\nTest label counts:")
print(test_df["LABEL"].value_counts())