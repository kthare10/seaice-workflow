#!/usr/bin/env python3

"""
Classify sea ice types from preprocessed ATL03 segments using a trained model.

Loads a trained LSTM or MLP model and runs inference on the full preprocessed
ATL03 dataset to classify each 2m segment as thick ice, thin ice, or open water.

Usage:
    python classify_seaice.py --input atl03_preprocessed.csv \
                               --model model.h5 \
                               --output classification_results.csv
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
CLASS_NAMES = {0: 'thick_ice', 1: 'thin_ice', 2: 'open_water'}
SEQUENCE_LENGTH = 10


def load_scaler(model_path):
    """
    Load scaler parameters saved during training.

    Args:
        model_path: Path to the model file (scaler saved with .scaler.npz suffix)

    Returns:
        Tuple of (mean, scale) arrays
    """
    import numpy as np

    scaler_file = str(Path(model_path).with_suffix('.scaler.npz'))
    try:
        data = np.load(scaler_file)
        return data['mean'], data['scale']
    except FileNotFoundError:
        logger.warning(f"Scaler file not found: {scaler_file}. Using default standardization.")
        return None, None


def prepare_lstm_sequences(X, seq_length=SEQUENCE_LENGTH):
    """
    Create sliding window sequences for LSTM inference.

    Args:
        X: Feature array (n_samples, n_features)
        seq_length: Window size

    Returns:
        X_seq (n_sequences, seq_length, n_features), valid indices
    """
    import numpy as np

    X_seq = []
    indices = []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        indices.append(i + seq_length - 1)
    return np.array(X_seq), np.array(indices)


def classify_seaice(input_file, model_path, output_file, batch_size=256):
    """
    Run sea ice classification on preprocessed ATL03 data.

    Args:
        input_file: Path to preprocessed ATL03 CSV
        model_path: Path to trained model file
        output_file: Path to output classification CSV
        batch_size: Inference batch size
    """
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    logger.info(f"Loading preprocessed data from {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Total segments: {len(df):,}")

    # Verify required columns
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        logger.error(f"Missing feature columns: {missing}")
        sys.exit(1)

    # Extract features
    X = df[FEATURE_COLUMNS].values

    # Load and apply scaler
    scaler_mean, scaler_scale = load_scaler(model_path)
    if scaler_mean is not None:
        X = (X - scaler_mean) / scaler_scale
    else:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'focal_loss_fn': lambda y_true, y_pred: y_pred}  # Placeholder for loading
    )
    model.summary(print_fn=logger.info)

    # Detect model type from input shape
    input_shape = model.input_shape
    is_lstm = len(input_shape) == 3  # (batch, seq_length, features)

    if is_lstm:
        logger.info("Detected LSTM model, preparing sequences")
        X_input, valid_indices = prepare_lstm_sequences(X)

        # Predict in batches
        predictions = model.predict(X_input, batch_size=batch_size, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        predicted_probs = np.max(predictions, axis=1)

        # Map predictions back to full dataframe
        df['predicted_class'] = -1
        df['prediction_prob'] = 0.0
        df.loc[valid_indices, 'predicted_class'] = predicted_classes
        df.loc[valid_indices, 'prediction_prob'] = predicted_probs

        # Fill edges with nearest valid prediction
        df['predicted_class'] = df['predicted_class'].replace(-1, method='bfill').replace(-1, method='ffill')
        df['prediction_prob'] = df['prediction_prob'].replace(0.0, method='bfill').replace(0.0, method='ffill')
    else:
        logger.info("Detected MLP model")
        predictions = model.predict(X, batch_size=batch_size, verbose=1)
        df['predicted_class'] = np.argmax(predictions, axis=1)
        df['prediction_prob'] = np.max(predictions, axis=1)

    # Add class name
    df['predicted_label'] = df['predicted_class'].map(CLASS_NAMES)

    # Save results
    df.to_csv(output_file, index=False)
    logger.info(f"Classification results saved to {output_file}")

    # Summary
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
                        help="Trained model file (HDF5)")
    parser.add_argument("--output", type=str, default="classification_results.csv",
                        help="Output classification CSV (default: classification_results.csv)")

    args = parser.parse_args()

    try:
        classify_seaice(args.input, args.model, args.output)
        logger.info("Classification completed successfully")
    except Exception as e:
        logger.error(f"Failed to classify sea ice: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
