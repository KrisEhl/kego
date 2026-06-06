# kego

## Pipeline
Usage idea:

Run training run by explicitly specifying params:
```
kego run --model catboost --params learning_rate:0.01 --hp-tune --hp-params max_trees::0:9:log
```

Run training run from config file.
```
kego run --config catboostv1

```
Run training run with overwritten config file params:
```
kego run --config catboostv1 --params featureset:v1
```

Following configuration files exist:
```yaml
# config.yaml
competition:
  - name: ragii
  - deadline: 2026.07.01

featurestore:
  - paths: [kristian@omarchyd, ~/]

models:
  - name: xgboost
    params:
      metric: auc

  - name: resnet
    params:
      n_layers: 10

ensemble:
  method: ridge_stack
```
