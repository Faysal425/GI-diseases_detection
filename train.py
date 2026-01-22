import argparse
import os
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, Sequential
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    Flatten,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    SeparableConv2D,
    GlobalAveragePooling2D,
    Reshape,
    Concatenate,
    multiply,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PSE-CNN on .npy image tensors")
    p.add_argument("--x_npy", type=str, required=True, help="Path to Stage1X.npy (images)")
    p.add_argument("--y_npy", type=str, required=True, help="Path to Stage1Y.npy (labels)")
    p.add_argument("--out_dir", type=str, default="outputs/psecnn", help="Output directory")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--test_size", type=float, default=0.10)
    p.add_argument("--val_size", type=float, default=0.10, help="Validation split from TRAIN portion")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--num_classes", type=int, default=3)
    return p.parse_args()


def squeeze_excite_block(x, ratio: int = 16):
    """Squeeze-and-Excitation (SE) block."""
    filters = int(x.shape[-1])
    se = GlobalAveragePooling2D()(x)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation="relu", kernel_initializer="he_normal", use_bias=False)(se)
    se = Dense(filters, activation="sigmoid", kernel_initializer="he_normal", use_bias=False)(se)
    return multiply([x, se])


def build_psecnn(input_shape: tuple[int, int, int], num_classes: int = 3) -> tf.keras.Model:
    """Build PSE-CNN with parallel separable conv branches + SE blocks."""
    inp = Input(shape=input_shape)

    parallel_kernels = [11, 9, 7, 5, 3]
    convs = []
    for k in parallel_kernels:
        conv = SeparableConv2D(256, k, padding="same", activation="relu")(inp)
        conv = squeeze_excite_block(conv)
        convs.append(conv)

    out = Concatenate()(convs)
    conv_model = Model(inputs=inp, outputs=out, name="psecnn_parallel_block")

    model = Sequential(name="PSE_CNN")
    model.add(conv_model)

    model.add(SeparableConv2D(128, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SeparableConv2D(64, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SeparableConv2D(32, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(SeparableConv2D(16, (3, 3), padding="same", name="lastconv"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(200, activation="relu", name="DenseLastPL"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation="softmax"))
    return model


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_plots(history_csv: Path, out_dir: Path) -> None:
    hist = pd.read_csv(history_csv)

    # Loss
    plt.figure()
    plt.plot(hist["loss"])
    if "val_loss" in hist.columns:
        plt.plot(hist["val_loss"])
        plt.legend(["train", "val"])
    else:
        plt.legend(["train"])
    plt.title("PSE-CNN Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=300)

    # Accuracy (Keras may log 'acc' or 'accuracy' depending on version)
    acc_key = "acc" if "acc" in hist.columns else "accuracy"
    val_acc_key = "val_acc" if "val_acc" in hist.columns else "val_accuracy"

    plt.figure()
    plt.plot(hist[acc_key])
    if val_acc_key in hist.columns:
        plt.plot(hist[val_acc_key])
        plt.legend(["train", "val"])
    else:
        plt.legend(["train"])
    plt.title("PSE-CNN Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy.png", dpi=300)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    # Load data
    X = np.load(args.x_npy)
    y = np.load(args.y_npy)

    # Basic checks
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
    if y.ndim != 1:
        y = y.reshape(-1)  # ensure (N,)
    if X.ndim != 4:
        raise ValueError(f"Expected X shape (N,H,W,C), got {X.shape}")

    # Split: train/test then train/val
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=args.val_size, stratify=y_train, random_state=args.seed
    )

    input_shape = X.shape[1:]
    model = build_psecnn(input_shape=input_shape, num_classes=args.num_classes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )
    model.summary()

    # Callbacks
    best_model_path = out_dir / "best_model.keras"
    history_csv = out_dir / "history.csv"

    ckpt = ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )
    csv_logger = CSVLogger(str(history_csv), separator=",", append=False)

    # Train
    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(X_val, y_val),
        callbacks=[ckpt, csv_logger],
        verbose=1,
    )
    elapsed = time.time() - start
    print(f"Training time (sec): {elapsed:.2f}")

    # Save final model
    final_model_path = out_dir / "final_model.keras"
    model.save(final_model_path)

    # Plots
    save_plots(history_csv=history_csv, out_dir=out_dir)

    # Evaluate on test
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # Save a small run summary
    summary_path = out_dir / "run_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"TF version: {tf.__version__}\n")
        f.write(f"Input shape: {input_shape}\n")
        f.write(f"Num classes: {args.num_classes}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"LR: {args.lr}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Train time (sec): {elapsed:.2f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write(f"Test acc: {test_acc:.4f}\n")

    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()