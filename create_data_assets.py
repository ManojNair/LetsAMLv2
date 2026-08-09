from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

folder_asset = Data(
    name="diabetes-folder",
    version="1",
    type=AssetTypes.URI_FOLDER,
    description="Folder containing diabetes CSVs.",
    path="data/",          # uploads the whole folder
)

table_asset = Data(
    name="diabetes-table",
    version="1",
    type=AssetTypes.MLTABLE,
    description="MLTable over diabetes.csv with typed columns.",
    path="data/diabetes-mltable/",   # folder containing the MLTable file
)

for asset in (folder_asset, table_asset):
    created = ml_client.data.create_or_update(asset)
    print(f"registered {created.name}:{created.version} ({created.type})")
