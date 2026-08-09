from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# DefaultAzureCredential tries, in order: environment variables, managed
# identity, Azure CLI login, interactive browser. On your laptop it will
# reuse your `az login` session.
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="08ec2468-4daa-430b-99fb-8871161bca5f",
    resource_group_name="LetsAMLRG",
    workspace_name="LetsAMLWS01"
)

ws = ml_client.workspaces.get(ml_client.workspace_name)
print(f"Connected to: {ws.name} ({ws.location})")
print(f"MLflow tracking URI: {ws.mlflow_tracking_uri}")
