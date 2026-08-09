"""Training script used by Labs 5 (command job), 7 (sweep), and 8 (pipeline).

Trains a logistic-regression classifier on the diabetes dataset and logs
everything to MLflow. Azure ML injects the tracking URI automatically when
this runs as a job, so the same script works locally and in the cloud.
"""
import argparse
import glob
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

FEATURES = [
    "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure", "TricepsThickness",
    "SerumInsulin", "BMI", "DiabetesPedigree", "Age",
]
TARGET = "Diabetic"


def load_data(path: str) -> pd.DataFrame:
    """Accept either a direct CSV path (uri_file) or a folder (uri_folder)."""
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {path}")
        return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return pd.read_csv(path)


def main(args):
    mlflow.autolog(log_models=False)  # params/metrics auto-logged; model logged explicitly below

    df = load_data(args.training_data)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=0, stratify=y
    )

    model = LogisticRegression(C=1 / args.reg_rate, solver="liblinear", max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_score)

    mlflow.log_param("reg_rate", args.reg_rate)
    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_auc", auc)
    print(f"accuracy={acc:.4f} auc={auc:.4f} (reg_rate={args.reg_rate})")

    # Log the model in MLflow format so it can be deployed without a scoring script.
    input_example = X_train.head(2)
    if args.model_output:
        mlflow.sklearn.save_model(model, args.model_output, input_example=input_example)
    else:
        mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", type=str, required=True,
                        help="Path to a CSV file or a folder of CSVs")
    parser.add_argument("--reg-rate", type=float, default=0.01,
                        help="Regularization rate (inverse of sklearn C)")
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--model-output", type=str, default=None,
                        help="Optional output folder for the model (used in pipelines)")
    main(parser.parse_args())
