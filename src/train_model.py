import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load preprocessed data
X_train = np.load("dataset_kaggle/X_train.npy")
y_train = np.load("dataset_kaggle/y_train.npy")
X_test = np.load("dataset_kaggle/X_test.npy")
y_test = np.load("dataset_kaggle/y_test.npy")

# Reshape for Conv1D: (samples, time_steps, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print("X_train reshaped:", X_train.shape)
print("X_test reshaped:", X_test.shape)

# --- Build the CNN ---
model = keras.Sequential([
    layers.Conv1D(16, kernel_size=5, activation="relu", input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(pool_size=4),

    layers.Conv1D(32, kernel_size=5, activation="relu"),
    layers.MaxPooling1D(pool_size=4),

    layers.Conv1D(64, kernel_size=5, activation="relu"),
    layers.MaxPooling1D(pool_size=4),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")   # single output: probability of "exoplanet"
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()







# Handle class imbalance: tell the model the minority class matters more
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("\nClass weights:", class_weight_dict)

# Stop early if validation performance stops improving (prevents overfitting/wasted time)
early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,      # hold out 20% of train data to monitor performance during training
    epochs=30,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

# Save the trained model for later use
model.save("exoplanet_cnn_model.keras")
print("\nModel saved as exoplanet_cnn_model.keras")