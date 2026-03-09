import json
import os

import numpy as np
import pandas as pd
from planqk.api.client import PlanqkApiClient

np.random.seed(42)
X_train = pd.DataFrame(
    {"f0": np.random.randn(5), "f1": np.random.randn(5), "f2": np.random.randn(5)}
)
y_train = pd.DataFrame({"target": [0, 1, 0, 1, 0]})
X_test = pd.DataFrame(
    {"f0": np.random.randn(5), "f1": np.random.randn(5), "f2": np.random.randn(5)}
)
y_test = pd.DataFrame({"target": [1, 0, 1, 0, 1]})

dataset = {
    "training_tabular_data": X_train.to_dict(),
    "training_target_data": y_train.to_dict(),
    "test_tabular_data": X_test.to_dict(),
    "test_target_data": y_test.to_dict(),
}

with open("data.json", "w") as f:
    json.dump(dataset, f)

api = PlanqkApiClient(
    access_token=os.environ["PLANQK_ACCESS_TOKEN"],
    organization_id=os.environ["PLANQK_ORGANIZATION_ID"],
)

input_dp = api.api.data_pools.create_data_pool(name="Rimay Bug Report Input")
with open("data.json", "rb") as f:
    api.api.data_pools.add_data_pool_file(id=input_dp.id, file=("data.json", f))

output_dp = api.api.data_pools.create_data_pool(name="Rimay Bug Report Output")
print(f"{output_dp=}")
from planqk.service.client import PlanqkServiceClient

client = PlanqkServiceClient(
    service_endpoint=os.environ["PLANQK_SERVICE_ENDPOINT"],
    access_key_id=os.environ["PLANQK_CONSUMER_KEY"],
    secret_access_key=os.environ["PLANQK_CONSUMER_SECRET"],
)

service_input = {
    "data": {
        "input_data_pool": {"id": input_dp.id, "ref": "DATAPOOL"},
        "output_data_pool": {"id": output_dp.id, "ref": "DATAPOOL"},
        "num_shots": 100,
        "num_runs": 1,
    }
}

execution = client.run(request=service_input)
print(f"{execution=}")
result = execution.result()  # Blocks until completion
print(f"{result=}")
