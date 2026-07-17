import numpy as np
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# --- Load trained model and test data ---
model = keras.models.load_model("exoplanet_cnn_model.keras")
X_test = np.load("dataset_kaggle/X_test.npy")
y_test = np.load("dataset_kaggle/y_test.npy")
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# --- Get predicted probabilities ---
y_pred_prob = model.predict(X_test)

# --- Inspect raw probabilities for the real planets vs. some non-planets ---
planet_indices = np.where(y_test == 1)[0]
print("Predicted probabilities for the 5 REAL planet stars:")
for i in planet_indices:
    print(f"  Star {i}: predicted probability = {y_pred_prob[i][0]:.4f}")

non_planet_indices = np.where(y_test == 0)[0][:10]
print("\nPredicted probabilities for 10 random NON-planet stars:")
for i in non_planet_indices:
    print(f"  Star {i}: predicted probability = {y_pred_prob[i][0]:.4f}")

# --- AUC-ROC: threshold-independent, computed once ---
auc = roc_auc_score(y_test, y_pred_prob)
print(f"\nAUC-ROC Score: {auc:.4f}  (this never changes — it's threshold-independent)")

# --- Evaluate at multiple thresholds to compare precision/recall trade-off ---
thresholds_to_try = [0.5, 0.3, 0.1, 0.05, 0.01]

for threshold in thresholds_to_try:
    y_pred = (y_pred_prob > threshold).astype(int)

    print(f"\n{'='*50}")
    print(f"--- Results at threshold = {threshold} ---")
    print(f"{'='*50}")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("(Rows = actual [0=no planet, 1=planet], Columns = predicted)")

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["No Planet", "Planet"],
        zero_division=0
    ))