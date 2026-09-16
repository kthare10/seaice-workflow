#!/usr/bin/env python3

"""
Classify sea ice types from preprocessed ATL03 segments using a trained model.

Loads the LSTM or MLP from train_model.py and labels every 2 m segment as
thick ice, thin ice, or open water. For the LSTM, windows are the same
centered 5-segment along-track windows used in training, built per track
(granule + beam); track ends are edge-padded so every segment is classified.

Usage:
    python classify_seaice.py --input atl03_preprocessed.csv \\
                               --model model.h5 \\
                               --output classification_results.csv
"""

import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Must match train_model.py
FEATURE_COLUMNS = ['mean_h', 'std_h', 'pcnth', 'd_pcnt', 'bcnt', 'd_brate']
NUM_CLASSES = 3
CLASS_NAMES = {0: 'thick_ice', 1: 'thin_ice', 2: 'open_water'}
SEQUENCE_LENGTH = 5


def load_scaler(model_path):
    """
    Load the StandardScaler parameters saved alongside the model.

    Args:
        model_path: Path to the model file (scaler saved with .scaler.npz suffix)

    Returns:
        Tuple of (mean, scale) arrays, or (None, None) if absent
    """
    import numpy as np

    scaler_file = str(Path(model_path).with_suffix('.scaler.npz'))
    try:
        data = np.load(scaler_file)
        return data['mean'], data['scale']
    except FileNotFoundError:
        # Classifying unscaled features with a model trained on standardized
        # ones silently produces garbage rather than failing, so refuse to run.
        logger.error(
            f"Scaler not found: {scaler_file}. train_model.py writes it next to the "
            "model and the features must be standardized with the same parameters "
            "used in training. Stage this file alongside the model."
        )
        sys.exit(1)


def centered_windows(X, seq_length=SEQUENCE_LENGTH):
    """
    Centered sliding windows over one track, edge-padded so every row has one.

    Args:
        X: Feature array (n_rows, n_features) in along-track order
        seq_length: Window length (odd)

    Returns:
        Array (n_rows, seq_length, n_features)
    """
    import numpy as np

    half = seq_length // 2
    padded = np.concatenate([np.repeat(X[:1], half, axis=0), X, np.repeat(X[-1:], half, axis=0)])
    return np.stack([padded[i:i + seq_length] for i in range(len(X))])


def classify_seaice(input_file, model_path, output_file, batch_size=256, granule=None):
    """
    Run sea ice classification on preprocessed ATL03 data.

    Args:
        input_file: Path to preprocessed ATL03 CSV
        model_path: Path to trained model file
        output_file: Path to output classification CSV
        batch_size: Inference batch size
        granule: If set, only rows of this granule are classified
    """
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU(s) available: {[g.name for g in gpus]}")
    else:
        logger.info("No GPU detected, using CPU")

    logger.info(f"Loading preprocessed data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Total segments: {len(df):,}")

    if granule is not None:
        if 'granule' not in df.columns:
            logger.error("--granule specified but input CSV has no 'granule' column")
            sys.exit(1)
        df = df[df['granule'] == granule].reset_index(drop=True)
        logger.info(f"Filtered to granule '{granule}': {len(df):,} segments")
        if len(df) == 0:
            logger.warning(f"No rows match granule '{granule}', writing empty output")
            pd.DataFrame(columns=pd.read_csv(input_file, nrows=0).columns.tolist()
                         + ['predicted_class', 'prediction_prob', 'predicted_label']
                         ).to_csv(output_file, index=False)
            return

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        logger.error(f"Missing feature columns: {missing}")
        sys.exit(1)

    scaler_mean, scaler_scale = load_scaler(model_path)

    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(
        model_path, compile=False,
        custom_objects={'focal_loss_fn': lambda y_true, y_pred: y_pred},
    )
    model.summary(print_fn=logger.info)
    is_lstm = len(model.input_shape) == 3
    logger.info(f"Detected {'LSTM' if is_lstm else 'MLP'} model")

    group_cols = [c for c in ('granule', 'beam') if c in df.columns]
    predicted_class = np.full(len(df), -1, dtype=int)
    predicted_prob = np.zeros(len(df))

    groups = df.groupby(group_cols, sort=False) if group_cols else [((None,), df)]
    for key, g in groups:
        g = g.sort_values('along_track_dist') if 'along_track_dist' in g.columns else g
        X = g[FEATURE_COLUMNS].values.astype(np.float32)
        ok = np.all(np.isfinite(X), axis=1)
        X = (X - scaler_mean) / scaler_scale
        X = np.nan_to_num(X, nan=0.0)

        if is_lstm:
            X_in = centered_windows(X)
        else:
            X_in = X
        probs = model.predict(X_in, batch_size=batch_size, verbose=0)
        cls = np.argmax(probs, axis=1)
        conf = np.max(probs, axis=1)
        cls[~ok] = -1
        conf[~ok] = 0.0
        predicted_class[g.index.values] = cls
        predicted_prob[g.index.values] = conf
        logger.info(f"  {key}: {len(g):,} segments classified")

    df['predicted_class'] = predicted_class
    df['prediction_prob'] = predicted_prob
    df['predicted_label'] = df['predicted_class'].map(CLASS_NAMES)

    df.to_csv(output_file, index=False)
    logger.info(f"Classification results saved to {output_file}")

    class_counts = df['predicted_class'].value_counts().sort_index()
    print(f"\n{'='*70}")
    print("SEA ICE CLASSIFICATION COMPLETE")
    print(f"{'='*70}")
    print(f"Total segments classified: {len(df):,}")
    print(f"Mean prediction confidence: {df['prediction_prob'].mean():.4f}")
    print("\nClass distribution:")
    for cls_val, count in class_counts.items():
        pct = 100 * count / len(df)
        print(f"  {CLASS_NAMES.get(cls_val, f'class_{cls_val}')}: {count:,} ({pct:.1f}%)")
    print(f"\nOutput: {output_file}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Classify sea ice types from ATL03 segments using trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input atl03_preprocessed.csv --model model.h5 --output classification_results.csv
        """
    )

    parser.add_argument("--input", type=str, required=True,
                        help="Input preprocessed ATL03 CSV file")
    parser.add_argument("--model", type=str, required=True,
                        help="Trained model file")
    parser.add_argument("--output", type=str, default="classification_results.csv",
                        help="Output classification CSV (default: classification_results.csv)")
    parser.add_argument("--granule", type=str, default=None,
                        help="Process only this granule (filter input by 'granule' column)")

    args = parser.parse_args()

    try:
        classify_seaice(args.input, args.model, args.output, granule=args.granule)
        logger.info("Classification completed successfully")
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Failed to classify sea ice: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
