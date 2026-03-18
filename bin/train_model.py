#!/usr/bin/env python3

"""
Train LSTM or MLP classifier for sea ice type classification.

Trains a deep learning model to classify ATL03 2m segments into three classes:
thick ice, thin ice, and open water. Supports LSTM (sequential) and MLP
(feed-forward) architectures with focal loss for class imbalance.

Based on: "Scalable Higher Resolution Polar Sea Ice Classification and Freeboard
Calculation from ICESat-2 ATL03 Data" (Iqrah et al., IPDPSW 2025)

Usage:
    python train_model.py --input labeled_data.csv \
                           --model-output model.h5 \
                           --metrics-output training_metrics.json \
                           --model-type lstm
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ['mean_h', 'median_h', 'std_h', 'photon_count', 'bg_rate']
NUM_CLASSES = 3
SEQUENCE_LENGTH = 10  # For LSTM: number of consecutive segments


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for handling class imbalance.

    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Balancing parameter (default: 0.25)

    Returns:
        Loss function
    """
    import tensorflow as tf

    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * y_true * tf.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
    return focal_loss_fn


def build_lstm_model(n_features, n_classes=NUM_CLASSES, seq_length=SEQUENCE_LENGTH):
    """
    Build LSTM model: 1 LSTM layer (16 units, ELU) + 7 Dense layers -> softmax.

    Args:
        n_features: Number of input features
        n_classes: Number of output classes
        seq_length: Sequence length for LSTM input

    Returns:
        Compiled Keras model
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(16, activation='elu',
                             input_shape=(seq_length, n_features)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(n_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy'],
    )

    return model


def build_mlp_model(n_features, n_classes=NUM_CLASSES):
    """
    Build MLP model: Dense(32, ReLU) -> Dense(3, softmax).

    Args:
        n_features: Number of input features
        n_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(n_features,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy'],
    )

    return model


def prepare_lstm_sequences(X, y, seq_length=SEQUENCE_LENGTH):
    """
    Create sliding window sequences for LSTM input.

    Args:
        X: Feature array (n_samples, n_features)
        y: Label array (n_samples,)
        seq_length: Window size

    Returns:
        X_seq (n_sequences, seq_length, n_features), y_seq (n_sequences,)
    """
    import numpy as np

    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length - 1])  # Label of the last segment in window
    return np.array(X_seq), np.array(y_seq)


def train_model(input_file, model_output, metrics_output, model_type="lstm",
                epochs=20, batch_size=64, test_size=0.2):
    """
    Train sea ice classification model.

    Args:
        input_file: Path to labeled data CSV
        model_output: Path to save model weights
        metrics_output: Path to save training metrics JSON
        model_type: Model type ('lstm' or 'mlp')
        epochs: Number of training epochs
        batch_size: Training batch size
        test_size: Fraction of data for testing
    """
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    logger.info(f"Loading labeled data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Total samples: {len(df):,}")

    # Verify required columns
    missing = [c for c in FEATURE_COLUMNS + ['label'] if c not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        sys.exit(1)

    # Extract features and labels
    X = df[FEATURE_COLUMNS].values
    y = df['label'].values.astype(int)

    logger.info(f"Features: {FEATURE_COLUMNS}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    # One-hot encode labels
    y_train_oh = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_oh = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    # Build model
    if model_type == "lstm":
        logger.info("Building LSTM model")
        X_train_seq, y_train_seq = prepare_lstm_sequences(X_train, y_train)
        X_test_seq, y_test_seq = prepare_lstm_sequences(X_test, y_test)
        y_train_seq_oh = tf.keras.utils.to_categorical(y_train_seq, NUM_CLASSES)
        y_test_seq_oh = tf.keras.utils.to_categorical(y_test_seq, NUM_CLASSES)

        model = build_lstm_model(n_features=len(FEATURE_COLUMNS))

        history = model.fit(
            X_train_seq, y_train_seq_oh,
            validation_data=(X_test_seq, y_test_seq_oh),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

        test_loss, test_acc = model.evaluate(X_test_seq, y_test_seq_oh, verbose=0)
    else:
        logger.info("Building MLP model")
        model = build_mlp_model(n_features=len(FEATURE_COLUMNS))

        history = model.fit(
            X_train, y_train_oh,
            validation_data=(X_test, y_test_oh),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

        test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)

    logger.info(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # Save model
    model.save(model_output)
    logger.info(f"Model saved to {model_output}")

    # Save scaler parameters alongside model
    scaler_file = str(Path(model_output).with_suffix('.scaler.npz'))
    np.savez(scaler_file, mean=scaler.mean_, scale=scaler.scale_)
    logger.info(f"Scaler saved to {scaler_file}")

    # Save training metrics
    metrics = {
        'model_type': model_type,
        'n_features': len(FEATURE_COLUMNS),
        'feature_names': FEATURE_COLUMNS,
        'n_classes': NUM_CLASSES,
        'epochs': epochs,
        'batch_size': batch_size,
        'test_size': test_size,
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'history': {
            'loss': [float(v) for v in history.history['loss']],
            'accuracy': [float(v) for v in history.history['accuracy']],
            'val_loss': [float(v) for v in history.history['val_loss']],
            'val_accuracy': [float(v) for v in history.history['val_accuracy']],
        },
        'scaler': {
            'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(),
        },
    }

    with open(metrics_output, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_output}")

    print(f"\n{'='*70}")
    print("MODEL TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Model type: {model_type.upper()}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Model saved: {model_output}")
    print(f"Metrics saved: {metrics_output}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train sea ice classification model (LSTM or MLP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train LSTM model
  %(prog)s --input labeled_data.csv --model-output model.h5 --model-type lstm

  # Train MLP model with custom epochs
  %(prog)s --input labeled_data.csv --model-output model.h5 --model-type mlp --epochs 50
        """
    )

    parser.add_argument("--input", type=str, required=True,
                        help="Input labeled data CSV file")
    parser.add_argument("--model-output", type=str, default="model.h5",
                        help="Output model file (default: model.h5)")
    parser.add_argument("--metrics-output", type=str, default="training_metrics.json",
                        help="Output metrics JSON file (default: training_metrics.json)")
    parser.add_argument("--model-type", type=str, choices=["lstm", "mlp"],
                        default="lstm", help="Model type (default: lstm)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size (default: 64)")

    args = parser.parse_args()

    try:
        train_model(
            input_file=args.input,
            model_output=args.model_output,
            metrics_output=args.metrics_output,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
