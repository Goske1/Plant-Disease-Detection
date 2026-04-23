"""
Plant Disease Detection — Model Training (TensorFlow / Keras)
BBM406 l5-ml-methodology:
  - Overfitting prevention: EarlyStopping + Dropout
  - Model selection via validation set (60 / 20 / 20 split)
  - Test set used exactly ONCE at the very end
"""

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# ── Hyperparameters ─────────────────────────────────────────────────────────────
IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 32      # samples processed per gradient update
NUM_EPOCHS    = 50      # one epoch = one full pass over the training set
LEARNING_RATE = 1e-3    # step size for the optimizer
NUM_CLASSES   = 15      # total disease classes across Tomato, Potato, Corn
DROPOUT_RATE  = 0.5     # fraction of neurons randomly disabled during training


# ── Data loading ────────────────────────────────────────────────────────────────
def load_datasets(data_root: str):
    """
    Reads train / val / test sub-folders produced by data_ingestion.py.
    Each sub-folder must follow: <split>/<crop>/<class>/<image>.jpg
    """
    common_kwargs = dict(
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        seed=42,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_root, "train"), **common_kwargs)
    val_ds   = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_root, "val"),   **common_kwargs)
    test_ds  = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_root, "test"),  **common_kwargs)

    # Normalize pixel values from [0, 255] to [0, 1]
    normalize = layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalize(x), y))
    val_ds   = val_ds.map(lambda x, y: (normalize(x), y))
    test_ds  = test_ds.map(lambda x, y: (normalize(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    return (
        train_ds.cache().prefetch(AUTOTUNE),
        val_ds.cache().prefetch(AUTOTUNE),
        test_ds.prefetch(AUTOTUNE),
    )


# ── CNN architecture ─────────────────────────────────────────────────────────────
def build_model(num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Three convolutional blocks followed by Global Average Pooling and Dropout.

    Conv layers learn hierarchical features (edges → textures → disease patterns).
    Dropout randomly zeros neurons during training — prevents overfitting
    (l5: small train error + large val error → model memorised instead of generalised).
    """
    model = models.Sequential(name="PlantDiseaseCNN")

    # Online augmentation — applied only during training, not during val/test
    model.add(layers.RandomFlip("horizontal_and_vertical"))
    model.add(layers.RandomRotation(0.15))
    model.add(layers.RandomZoom(0.1))

    # Block 1 — detect low-level features: edges, colour patches
    model.add(layers.Conv2D(32, kernel_size=3, padding="same",
                            activation="relu", input_shape=(*IMAGE_SIZE, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2))

    # Block 2 — combine low-level features into textures
    model.add(layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2))

    # Block 3 — abstract patterns specific to each disease class
    model.add(layers.Conv2D(128, kernel_size=3, padding="same", activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=2))

    # Global Average Pooling replaces a large Flatten+Dense block
    # Each feature map is reduced to its mean → far fewer parameters
    model.add(layers.GlobalAveragePooling2D())

    # Dropout: at each training step, DROPOUT_RATE fraction of activations is zeroed
    # Forces the network to learn redundant representations → better generalisation
    model.add(layers.Dropout(DROPOUT_RATE))

    # Output layer: softmax converts raw scores to class probabilities
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


# ── Training ─────────────────────────────────────────────────────────────────────
def train(data_root: str, output_dir: str = "models/run") -> dict:
    os.makedirs(output_dir, exist_ok=True)

    train_ds, val_ds, test_ds = load_datasets(data_root)
    model = build_model()

    # Optimizer minimises the loss via gradient descent.
    # Adam adapts the learning rate per parameter — generally converges faster than SGD.
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",   # log-loss for multi-class classification
        metrics=["accuracy"],
    )
    model.summary()

    # EarlyStopping watches val_loss — stops training if it doesn't improve for
    # `patience` consecutive epochs and restores the best weights found so far.
    # This directly addresses the overfitting problem described in l5-ml-methodology.
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    # Save the best checkpoint to disk (lowest val_loss)
    checkpoint = callbacks.ModelCheckpoint(
        filepath=os.path.join(output_dir, "best_model.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=0,
    )

    history = model.fit(
        train_ds,
        epochs=NUM_EPOCHS,
        validation_data=val_ds,     # used for model selection only, never for weight updates
        callbacks=[early_stop, checkpoint],
    )

    # ── Final test evaluation ────────────────────────────────────────────────────
    # Per l5-ml-methodology: the test set is used ONCE to report final performance.
    # It must not influence any training or tuning decision.
    print("\n" + "─" * 52)
    print("  Final Test Evaluation (run once — not used for tuning)")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f}  ({test_acc * 100:.2f}%)")
    print("─" * 52 + "\n")

    plot_history(history, output_dir)

    return {
        "history":   history.history,
        "test_loss": test_loss,
        "test_acc":  test_acc,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────────
def plot_history(history, output_dir: str) -> None:
    """
    Draw Accuracy and Loss curves for training and validation splits.

    A large gap between train and val curves signals overfitting.
    EarlyStopping should prevent the curves from diverging past the optimal point.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    epochs = range(1, len(history.history["accuracy"]) + 1)

    # Accuracy
    axes[0].plot(epochs, history.history["accuracy"],     label="Train")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Loss — gap between train_loss and val_loss = degree of overfitting
    axes[1].plot(epochs, history.history["loss"],     label="Train")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (CrossEntropy)")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("Plant Disease CNN — Training Curves", fontsize=13, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved → {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train plant disease CNN with Keras.")
    parser.add_argument(
        "--data",
        default=r"C:\Users\gokse\yeni-proje\dataset\split",
        help="Root of split dataset (train/val/test subfolders)",
    )
    parser.add_argument(
        "--output",
        default=r"C:\Users\gokse\yeni-proje\models\run",
        help="Directory for checkpoints and plots",
    )
    args = parser.parse_args()

    train(args.data, args.output)
