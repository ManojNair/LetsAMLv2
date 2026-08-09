import mlflow
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
mlflow.set_tracking_uri(
    ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
)

runs = mlflow.search_runs(
    experiment_names=["diabetes-training"],
)

metric_columns = ["metrics.test_auc", "metrics.test_accuracy"]
missing_metrics = [column for column in metric_columns if column not in runs.columns]
if missing_metrics:
    raise RuntimeError(
        "No completed training runs with test metrics were found. "
        "Wait for a job to complete, then run this script again."
    )

runs = runs.dropna(subset=metric_columns).copy()
if runs.empty:
    raise RuntimeError("No completed training runs contain test metrics.")
runs = runs.sort_values("metrics.test_auc", ascending=False)

if "params.reg_rate" not in runs.columns:
    runs["params.reg_rate"] = "not logged"
else:
    runs["params.reg_rate"] = runs["params.reg_rate"].fillna("not logged")

columns = ["run_id", "params.reg_rate", *metric_columns]
print(runs[columns].to_string(index=False))
best = runs.iloc[0]
print(f"\nBest run: {best.run_id} (AUC {best['metrics.test_auc']:.4f})")
