"""Data-preparation step for the pipeline in Lab 8.

Reads raw CSV data, removes rows with missing values, normalizes the numeric
feature columns, and writes the cleaned data to the output folder.
"""
import argparse
import glob
import os

import pandas as pd

FEATURES = [
    "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure", "TricepsThickness",
    "SerumInsulin", "BMI", "DiabetesPedigree", "Age",
]


def main(args):
    if os.path.isdir(args.input_data):
        files = glob.glob(os.path.join(args.input_data, "*.csv"))
        df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    else:
        df = pd.read_csv(args.input_data)

    n_before = len(df)
    df = df.dropna()
    print(f"dropped {n_before - len(df)} rows with missing values")

    # Min-max scale features; keep the label and ID untouched.
    for col in FEATURES:
        lo, hi = df[col].min(), df[col].max()
        df[col] = (df[col] - lo) / (hi - lo)

    os.makedirs(args.output_data, exist_ok=True)
    df.to_csv(os.path.join(args.output_data, "prepped.csv"), index=False)
    print(f"wrote {len(df)} rows to {args.output_data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    parser.add_argument("--output-data", type=str, required=True)
    main(parser.parse_args())
