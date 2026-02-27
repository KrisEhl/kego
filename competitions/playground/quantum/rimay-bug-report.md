# Bug Report: Rimay Quantum Feature Extraction Simulator — All Executions Fail Silently

## Summary

All service executions of the **Rimay - Quantum Feature Extraction - Simulator** fail with status `FAILED` immediately (~30s), with no error message, no logs, and no output files — regardless of input data size or format.

## Service

- **Service**: Rimay - Quantum Feature Extraction - Simulator
- **Service Definition ID**: `c486bb09-7827-4324-a655-6ddb135d0ed9`
- **Service ID**: `f56e9b97-3bc1-4c8d-99b9-1c2d0b2f9944`
- **Application ID**: `9d5b6da0-b782-4208-bae4-613ac512e051`
- **Endpoint**: `https://gateway.hub.kipu-quantum.com/kipu-quantum/rimay---quantum-feature-extraction---simulator/1.0.0`

## Environment

- **planqk-service-sdk**: 2.12.0
- **planqk-api-sdk**: 1.9.3
- **planqk-quantum**: 3.1.1
- **Python**: 3.13.12
- **OS**: macOS 26.3 (Apple Silicon)

## Steps to Reproduce

### 1. Create input data and upload to datapool

```python
import pandas as pd, numpy as np, json, os
from planqk.api.client import PlanqkApiClient

np.random.seed(42)
X_train = pd.DataFrame({"f0": np.random.randn(5), "f1": np.random.randn(5), "f2": np.random.randn(5)})
y_train = pd.DataFrame({"target": [0, 1, 0, 1, 0]})
X_test = pd.DataFrame({"f0": np.random.randn(5), "f1": np.random.randn(5), "f2": np.random.randn(5)})
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
```

### 2. Submit to Rimay service

```python
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
result = execution.result()  # Blocks until completion
```

### 3. Observe failure

Execution status transitions: `RUNNING` → `FAILED` (~30 seconds)

## What I Tried

| Attempt | Input Data | Request Format | Result |
|---------|-----------|----------------|--------|
| 1 | 699 train + 299 test, 13 features, StandardScaler normalized | `{"data": {all fields}}` | FAILED |
| 2 | 699 train + 299 test, 13 features, raw unscaled | `{"data": {all fields}}` | FAILED |
| 3 | 5 train + 5 test, 3 features, random floats | `{"data": {all fields}}` | FAILED |
| 4 | 5 train + 5 test, 3 features, random floats | `{"data": {pools}, "params": {shots, runs}}` | FAILED |

All within free-tier limits (≤15 features, ≤1000 samples).

## Verification

- **Datapool upload confirmed**: Downloaded file from input datapool, verified correct structure (`data.json` with 4 required keys, correct shapes)
- **Authentication works**: Datapool creation/upload succeeds; service execution starts (gets RUNNING status)
- **No logs available**: `GET /{execution_id}/logs` returns 404 for WORKFLOW type executions
- **No output files**: Output datapool remains empty after execution

## Failed Execution IDs

- `0873bb69-d979-4614-99af-7402ffd2fd16` (2026-02-27 10:04:50)
- `06cfeeef-2ee4-4985-ac7c-22a0dbfc172a` (2026-02-27 10:06:41)
- `1f897d64-29ca-4245-959e-b61545e97698` (2026-02-27 10:07:54)
- `2ad6150f-ff38-451c-b1d3-007182ca6984` (2026-02-27 10:09:55)

## Expected Behavior

Execution should complete with status `SUCCEEDED` and output `.npy` files in the output datapool:
- `1_Xq_train_0.npy` — quantum features for training set
- `1_Xq_validation_0.npy` — quantum features for test set
- `1_yq_train_0.npy` / `1_yq_validation_0.npy` — target labels

## Request

Could you check the internal workflow logs for the failed executions above? The service fails silently — no error message is returned to the client, and the log endpoint returns 404 for WORKFLOW-type executions.
