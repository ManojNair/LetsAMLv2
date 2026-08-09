# read_table.py — materialize the mltable asset into pandas
import mltable
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
asset = ml_client.data.get("diabetes-table", version="1")

tbl = mltable.load(asset.path)     # loads the MLTable blueprint
df = tbl.to_pandas_dataframe()     # materializes it
print(df.dtypes)
print(df.head())
