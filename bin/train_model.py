#!/usr/bin/env python3

"""
Train the LSTM or MLP sea-ice classifier of Iqrah et al. (IPDPSW 2025).

Sec. III.B / IV.A of the paper, as implemented here:

  * six features per 2 m segment: height (mean_h), height standard deviation
    (std_h), high-confidence photon rate (pcnth), photon-rate change (d_pcnt),
    background photon count (bcnt), background-rate change (d_brate);
  * LSTM input is a 5-segment window centered on the segment being classified
    (the paper: "classifying a point in nth position depends on ... n-2, n-1,
    n+1 and n+2"); windows are built per track (granule + beam) in along-track
    order BEFORE the train/test split, so every sequence is a real stretch of
    track;
  * LSTM(16, ELU, dropout 0.2) -> Dense(32, 96, 32, 16, 112, 48, 64; ELU)
    -> Dense(3, softmax);
  * MLP: Dense(32, ReLU) -> Dropout(0.2) -> Dense(3, softmax);
  * Adam, learning rate 0.003, focal loss, batch size 32, 20 epochs, 80/20
    stratified split;
  * accuracy, precision, recall, F1 and the confusion matrix on the 20%
    held-out set are written to the metrics JSON (paper Table III / Fig. 4).

Usage:
    python train_model.py --input labeled_data.csv \\
                           --model-output model.h5 \\
                           --metrics-output training_metrics.json \\
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

# Paper Sec. III.B.1: "height/elevation, height standard deviation,
# high-confidence photon, photon rate changes, background photon, and
# background photon rate changes"
FEATURE_COLUMNS = ['mean_h', 'std_h', 'pcnth', 'd_pcnt', 'bcnt', 'd_brate']
NUM_CLASSES = 3
CLASS_NAMES = {0: 'thick_ice', 1: 'thin_ice', 2: 'open_water'}
SEQUENCE_LENGTH = 5       # n-2 .. n+2
SEQUENCE_CENTER = 2
LEARNING_RATE = 0.003
BATCH_SIZE = 32
EPOCHS = 20
TEST_SIZE = 0.2
LSTM_UNITS = 16
LSTM_DROPOUT = 0.2
DENSE_UNITS = [32, 96, 32, 16, 112, 48, 64]
MLP_UNITS = 32
MLP_DROPOUT = 0.2


def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss (Lin et al. 2017) for class imbalance.

    The paper states focal loss is used but gives no gamma/alpha; these are
    the common defaults. Note alpha is a single scalar applied to every class,
    so it scales the loss rather than re-weighting minority classes.

    Args:
        gamma: Focusing parameter
        alpha: Balancing parameter

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
    Build the paper's LSTM: LSTM(16, ELU, dropout 0.2) + 7 ELU Dense layers + softmax.

    Args:
        n_features: Number of input features
        n_classes: Number of output classes
        seq_length: Sequence length for LSTM input

    Returns:
        Compiled Keras model
    """
    import tensorflow as tf

    layers = [
        tf.keras.Input(shape=(seq_length, n_features)),
        tf.keras.layers.LSTM(LSTM_UNITS, activation='elu', dropout=LSTM_DROPOUT),
    ]
    for units in DENSE_UNITS:
        layers.append(tf.keras.layers.Dense(units, activation='elu'))
    layers.append(tf.keras.layers.Dense(n_classes, activation='softmax'))
    model = tf.keras.Sequential(layers)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=focal_loss(),
        metrics=['accuracy'],
    )
    return model


def build_mlp_model(n_features, n_classes=NUM_CLASSES):
    """
    Build the paper's MLP: Dense(32, ReLU) -> Dropout(0.2) -> Dense(3, softmax).

    Args:
        n_features: Number of input features
        n_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(n_features,)),
        tf.keras.layers.Dense(MLP_UNITS, activation='relu'),
        tf.keras.layers.Dropout(MLP_DROPOUT),
        tf.keras.layers.Dense(n_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=focal_loss(),
        metrics=['accuracy'],
    )
    return model


def build_track_sequences(df, seq_length=SEQUENCE_LENGTH):
    """
    Build centered along-track windows, one per interior segment of each track.

    Tracks are (granule, beam) groups sorted by along_track_dist. A window for
    segment n spans n-2 .. n+2 and is labeled with segment n's label, so no
    window crosses a track boundary and the order is the real along-track
    order.

    Args:
        df: Labeled DataFrame with FEATURE_COLUMNS, label, granule, beam,
            along_track_dist
        seq_length: Window length (odd)

    Returns:
        X_seq (n_windows, seq_length, n_features), y (n_windows,),
        center_rows (n_windows,) row indices into df of the window centers
    """
    import numpy as np

    half = seq_length // 2
    X_seq, y_seq, centers = [], [], []
    group_cols = [c for c in ('granule', 'beam') if c in df.columns]
    groups = df.groupby(group_cols, sort=False) if group_cols else [((None,), df)]

    for _, g in groups:
        g = g.sort_values('along_track_dist')
        if len(g) < seq_length:
            continue
        X = g[FEATURE_COLUMNS].values.astype(np.float32)
        y = g['label'].values.astype(int)
        idx = g.index.values
        for i in range(half, len(g) - half):
            X_seq.append(X[i - half:i + half + 1])
            y_seq.append(y[i])
            centers.append(idx[i])

    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq), np.asarray(centers)


def evaluate(y_true, y_pred):
    """
    Accuracy, per-class and macro precision/recall/F1, and confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        JSON-serializable dict
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    labels = list(range(NUM_CLASSES))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        'n_samples': int(len(y_true)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_macro': float(p_m), 'recall_macro': float(r_m), 'f1_macro': float(f_m),
        'precision_weighted': float(p_w), 'recall_weighted': float(r_w), 'f1_weighted': float(f_w),
        'per_class': {
            CLASS_NAMES[c]: {'precision': float(p[c]), 'recall': float(r[c]),
                             'f1': float(f[c]), 'support': int(s[c])}
            for c in labels
        },
        'confusion_matrix': cm.tolist(),
        'confusion_matrix_labels': [CLASS_NAMES[c] for c in labels],
        'class_distribution': {CLASS_NAMES[c]: int(np.sum(y_true == c)) for c in labels},
    }


def train_model(input_file, model_output, metrics_output, model_type="lstm",
                epochs=EPOCHS, batch_size=BATCH_SIZE, test_size=TEST_SIZE):
    """
    Train the sea-ice classification model.

    Args:
        input_file: Path to labeled data CSV
        model_output: Path to save the model
        metrics_output: Path to save the training metrics JSON
        model_type: 'lstm' or 'mlp'
        epochs: Number of training epochs
        batch_size: Training batch size
        test_size: Held-out fraction
    """
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU(s) available: {[g.name for g in gpus]}")
    else:
        logger.info("No GPU detected, using CPU")

    logger.info(f"Loading labeled data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Total samples: {len(df):,}")

    missing = [c for c in FEATURE_COLUMNS + ['label', 'along_track_dist'] if c not in df.columns]
    if missing:
        logger.error(f"Missing columns: {missing}")
        sys.exit(1)

    df = df[df['label'].isin(range(NUM_CLASSES))].dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    y_all = df['label'].values.astype(int)
    logger.info(f"Features: {FEATURE_COLUMNS}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y_all, return_counts=True)))}")

    # Sequences are built per track in along-track order, then split; the
    # scaler is fit on training data only
    X_seq, y_seq, centers = build_track_sequences(df)
    logger.info(f"Track windows of {SEQUENCE_LENGTH} segments: {len(y_seq):,} "
                f"(from {df.groupby(['granule', 'beam']).ngroups} tracks)")
    if len(y_seq) == 0:
        logger.error("No track has enough segments to form a window")
        sys.exit(1)

    idx_train, idx_test = train_test_split(
        np.arange(len(y_seq)), test_size=test_size, random_state=42, stratify=y_seq
    )
    logger.info(f"Train windows: {len(idx_train):,}, Test windows: {len(idx_test):,}")

    scaler = StandardScaler().fit(X_seq[idx_train].reshape(-1, len(FEATURE_COLUMNS)))

    def scale(X):
        shape = X.shape
        return scaler.transform(X.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

    if model_type == "lstm":
        X_train, X_test = scale(X_seq[idx_train]), scale(X_seq[idx_test])
    else:
        # MLP classifies each segment from its own features (window center)
        X_train = scale(X_seq[idx_train][:, SEQUENCE_CENTER, :])
        X_test = scale(X_seq[idx_test][:, SEQUENCE_CENTER, :])
    y_train, y_test = y_seq[idx_train], y_seq[idx_test]
    y_train_oh = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_oh = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    if model_type == "lstm":
        logger.info("Building LSTM model")
        model = build_lstm_model(n_features=len(FEATURE_COLUMNS))
    else:
        logger.info("Building MLP model")
        model = build_mlp_model(n_features=len(FEATURE_COLUMNS))
    model.summary(print_fn=logger.info)

    history = model.fit(
        X_train, y_train_oh,
        validation_data=(X_test, y_test_oh),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    y_pred = np.argmax(model.predict(X_test, batch_size=256, verbose=0), axis=1)
    held_out = evaluate(y_test, y_pred)
    logger.info(
        f"Held-out: accuracy={held_out['accuracy']:.4f} "
        f"precision={held_out['precision_macro']:.4f} recall={held_out['recall_macro']:.4f} "
        f"f1={held_out['f1_macro']:.4f}"
    )

    model.save(model_output)
    logger.info(f"Model saved to {model_output}")

    scaler_file = str(Path(model_output).with_suffix('.scaler.npz'))
    np.savez(scaler_file, mean=scaler.mean_, scale=scaler.scale_)
    logger.info(f"Scaler saved to {scaler_file}")

    metrics = {
        'model_type': model_type,
        'features': FEATURE_COLUMNS,
        'n_features': len(FEATURE_COLUMNS),
        'n_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'sequence_length': SEQUENCE_LENGTH if model_type == "lstm" else 1,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': LEARNING_RATE,
        'loss': 'focal(gamma=2.0, alpha=0.25)',
        'test_size': test_size,
        'n_tracks': int(df.groupby(['granule', 'beam']).ngroups),
        'train_samples': int(len(idx_train)),
        'test_samples': int(len(idx_test)),
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'held_out': held_out,
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()},
        'scaler': {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()},
    }
    with open(metrics_output, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_output}")

    print(f"\n{'='*70}")
    print("MODEL TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Model type: {model_type.upper()}")
    print(f"Windows: {len(y_seq):,} from {metrics['n_tracks']} tracks  (train {len(idx_train):,} / test {len(idx_test):,})")
    print(f"Held-out accuracy: {held_out['accuracy']:.4f}  precision: {held_out['precision_macro']:.4f}  "
          f"recall: {held_out['recall_macro']:.4f}  F1: {held_out['f1_macro']:.4f}")
    for name, m in held_out['per_class'].items():
        print(f"  {name:11s} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} n={m['support']}")
    print(f"Model saved: {model_output}")
    print(f"Metrics saved: {metrics_output}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train sea ice classification model (LSTM or MLP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input labeled_data.csv --model-output model.h5 --model-type lstm
  %(prog)s --input labeled_data.csv --model-output model.h5 --model-type mlp --epochs 40
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
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS}, paper Sec. IV.A)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Training batch size (default: {BATCH_SIZE}, paper Sec. IV.A)")

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
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Failed to train model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
