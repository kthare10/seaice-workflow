#!/usr/bin/env python3

"""
Merge per-granule classification CSVs into a single file.

Usage:
    python merge_classifications.py --inputs cls_0.csv cls_1.csv --output classification_results.csv
"""

import argparse
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Merge per-granule classification CSVs")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input CSV files")
    parser.add_argument("--output", required=True, help="Merged output CSV")
    args = parser.parse_args()

    frames = []
    for path in args.inputs:
        df = pd.read_csv(path)
        if len(df) > 0:
            frames.append(df)

    if frames:
        merged = pd.concat(frames, ignore_index=True)
    else:
        # All inputs were empty; preserve headers from first file
        merged = pd.read_csv(args.inputs[0], nrows=0)

    merged.to_csv(args.output, index=False)
    print(f"Merged {len(frames)} non-empty files -> {len(merged):,} rows -> {args.output}")


if __name__ == "__main__":
    main()
