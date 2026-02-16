RAY_HEAD_IP ?= 192.168.178.32
RAY_HEAD_PORT ?= 6379
RAY_DASHBOARD ?= http://$(RAY_HEAD_IP):8265
MLFLOW_PORT ?= 5000
MLFLOW_DIR ?= $(HOME)/mlflow
MLFLOW_TRACKING_URI ?= http://$(RAY_HEAD_IP):$(MLFLOW_PORT)

start-head:
	RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 uv run ray start --head --port=$(RAY_HEAD_PORT) --node-ip-address $(RAY_HEAD_IP) --dashboard-host=0.0.0.0 --dashboard-port=8265 --ray-client-server-port=10001 --num-cpus=$$(expr $$(nproc --all) - 2) --resources '{"heavy_gpu": 2}'

start-worker:
	$(eval NODE_IP := $(shell if grep -qi microsoft /proc/version 2>/dev/null; then ip -4 addr show | grep -oP 'inet 192\.168\.\d+\.\d+' | head -1 | grep -oP '192\.168\.\d+\.\d+'; fi))
	RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 uv run ray start --address="$(RAY_HEAD_IP):$(RAY_HEAD_PORT)" $(if $(NODE_IP),--node-ip-address=$(NODE_IP)) --resources '{"heavy_gpu": 1}'

start-worker-light:
	RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 uv run ray start --address="$(RAY_HEAD_IP):$(RAY_HEAD_PORT)" --resources '{"light_gpu": 1}'

restart-worker:
	uv run ray stop --force
	$(MAKE) start-worker

stop:
	uv run ray stop --force

status:
	@uv run ray job list --address $(RAY_DASHBOARD) 2>/dev/null | python3 -c "\
	import sys, re; \
	text = sys.stdin.read(); \
	jobs = re.findall(r\"submission_id='(raysubmit_\w+)'.*?status=<JobStatus\.(\w+):.*?entrypoint='([^']+)'\", text, re.DOTALL); \
	[print(f'{sid}  {status:<10s}  {cmd}') for sid, status, cmd in jobs]"

logs:
	@JOB_ID=$(if $(JOB),$(JOB),$$(uv run ray job list --address $(RAY_DASHBOARD) 2>/dev/null | \
		python3 -c "import sys,re; text=sys.stdin.read(); \
		jobs=re.findall(r\"submission_id='(raysubmit_\w+)'.*?status=<JobStatus\.(\w+):\",text,re.DOTALL); \
		running=[j for j,s in jobs if s=='RUNNING']; \
		print(running[-1] if running else '')")); \
	if [ -z "$$JOB_ID" ]; then echo "No running jobs"; \
	else uv run ray job logs "$$JOB_ID" --address $(RAY_DASHBOARD) 2>/dev/null | \
		python3 scripts/train_logs.py "$$JOB_ID"; \
	fi

N ?= 20
logs-raw:
	@JOB_ID=$(if $(JOB),$(JOB),$$(uv run ray job list --address $(RAY_DASHBOARD) 2>/dev/null | \
		python3 -c "import sys,re; text=sys.stdin.read(); \
		jobs=re.findall(r\"submission_id='(raysubmit_\w+)'.*?status=<JobStatus\.(\w+):\",text,re.DOTALL); \
		running=[j for j,s in jobs if s=='RUNNING']; \
		print(running[-1] if running else '')")); \
	if [ -z "$$JOB_ID" ]; then echo "No running jobs"; \
	else uv run ray job logs "$$JOB_ID" --address $(RAY_DASHBOARD) 2>/dev/null | tail -n $(N); \
	fi

mlflow-start:
	mkdir -p $(MLFLOW_DIR)/artifacts
	nohup uv run mlflow server \
		--backend-store-uri sqlite:///$(MLFLOW_DIR)/mlflow.db \
		--default-artifact-root mlflow-artifacts:/ \
		--artifacts-destination $(MLFLOW_DIR)/artifacts \
		--host 0.0.0.0 --port $(MLFLOW_PORT) \
		> $(MLFLOW_DIR)/server.log 2>&1 & echo $$! > $(MLFLOW_DIR)/server.pid

mlflow-stop:
	@if [ -f $(MLFLOW_DIR)/server.pid ]; then \
		kill $$(cat $(MLFLOW_DIR)/server.pid) 2>/dev/null; \
		rm -f $(MLFLOW_DIR)/server.pid; \
		echo "MLflow server stopped"; \
	else \
		echo "No PID file found"; \
	fi

log-score:
	@test -n "$(SCORE)" || (echo "Usage: make log-score SCORE=0.953"; exit 1)
	@uv run python3 -c "\
	import mlflow; \
	mlflow.set_tracking_uri('$(MLFLOW_TRACKING_URI)'); \
	runs = mlflow.search_runs(search_all_experiments=True, filter_string=\"run_name LIKE 'ensemble_%'\", order_by=['start_time DESC'], max_results=1); \
	assert len(runs) > 0, 'No ensemble runs found'; \
	rid = runs.iloc[0].run_id; \
	name = runs.iloc[0]['tags.mlflow.runName']; \
	exp = mlflow.get_experiment(runs.iloc[0].experiment_id).name; \
	client = mlflow.tracking.MlflowClient(); \
	client.log_metric(rid, 'public_lb_score', $(SCORE)); \
	print(f'Logged public_lb_score=$(SCORE) to run \"{name}\" in experiment \"{exp}\"')"

submit-fast:
	uv run ray job submit --address $(RAY_DASHBOARD) --no-wait -- bash -c 'export KEGO_PATH_DATA=/home/kristian/projects/kego/data && export PYTHONUNBUFFERED=1 && export MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) && python ../notebooks/playground/train_s6e2_baseline.py --fast $(if $(TAG),--tag $(TAG))'

submit-fast-full:
	uv run ray job submit --address $(RAY_DASHBOARD) --no-wait -- bash -c 'export KEGO_PATH_DATA=/home/kristian/projects/kego/data && export PYTHONUNBUFFERED=1 && export MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) && python ../notebooks/playground/train_s6e2_baseline.py --fast-full $(if $(TAG),--tag $(TAG)) $(if $(FEATURES),--features $(FEATURES))'

submit-full:
	uv run ray job submit --address $(RAY_DASHBOARD) --no-wait -- bash -c 'export KEGO_PATH_DATA=/home/kristian/projects/kego/data && export PYTHONUNBUFFERED=1 && export MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) && python ../notebooks/playground/train_s6e2_baseline.py $(if $(TAG),--tag $(TAG)) $(if $(FEATURES),--features $(FEATURES))'

submit-neural:
	uv run ray job submit --address $(RAY_DASHBOARD) --no-wait -- bash -c 'export KEGO_PATH_DATA=/home/kristian/projects/kego/data && export PYTHONUNBUFFERED=1 && export MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) && python ../notebooks/playground/train_s6e2_baseline.py --neural $(if $(TAG),--tag $(TAG))'

submit-debug:
	uv run ray job submit --address $(RAY_DASHBOARD) --no-wait -- bash -c 'export KEGO_PATH_DATA=/home/kristian/projects/kego/data && export PYTHONUNBUFFERED=1 && export MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) && python ../notebooks/playground/train_s6e2_baseline.py --debug --fast $(if $(TAG),--tag $(TAG))'

submit-ensemble:
	@test -n "$(ENSEMBLE)$(EXPERIMENTS)" || (echo "Usage: make submit-ensemble ENSEMBLE=submit-v1"; echo "       make submit-ensemble EXPERIMENTS='exp1 exp2'"; exit 1)
	uv run ray job submit --address $(RAY_DASHBOARD) --no-wait -- bash -c 'export KEGO_PATH_DATA=/home/kristian/projects/kego/data && export PYTHONUNBUFFERED=1 && export MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) && python ../notebooks/playground/train_s6e2_baseline.py $(if $(ENSEMBLE),--from-ensemble $(ENSEMBLE),--from-experiment $(EXPERIMENTS))'

submit-kaggle:
	@test -n "$(ENSEMBLE)$(EXPERIMENTS)" || (echo "Usage: make submit-kaggle ENSEMBLE=submit-v1"; echo "       make submit-kaggle EXPERIMENTS='exp1 exp2'"; exit 1)
	uv run ray job submit --address $(RAY_DASHBOARD) --no-wait -- bash -c 'export KEGO_PATH_DATA=/home/kristian/projects/kego/data && export PYTHONUNBUFFERED=1 && export MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) && python ../notebooks/playground/train_s6e2_baseline.py --submit $(if $(ENSEMBLE),--from-ensemble $(ENSEMBLE),--from-experiment $(EXPERIMENTS))'

promote:
	@test -n "$(ENSEMBLE)" || (echo "Usage: make promote ENSEMBLE=submit-v1 RUN_ID='id1 id2'"; exit 1)
	@test -n "$(RUN_ID)" || (echo "Usage: make promote ENSEMBLE=submit-v1 RUN_ID='id1 id2'"; exit 1)
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) uv run python3 scripts/promote.py add $(ENSEMBLE) --run-id $(RUN_ID)

auto-promote:
	@test -n "$(ENSEMBLE)" || (echo "Usage: make auto-promote ENSEMBLE=submit-v1 EXPERIMENT='full-v1 gbdt-v1' [FOLDS=10]"; echo "       make auto-promote ENSEMBLE=submit-v1 ALL=1 [FOLDS=10]"; exit 1)
	@test -n "$(EXPERIMENT)$(ALL)" || (echo "Error: set EXPERIMENT='...' or ALL=1"; exit 1)
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) uv run python3 scripts/promote.py auto $(ENSEMBLE) $(if $(ALL),--all,--experiment $(EXPERIMENT)) $(if $(FOLDS),--folds $(FOLDS)) $(if $(MODELS),--model $(MODELS)) $(if $(FEATURES),--features $(FEATURES))

list-ensemble:
	@test -n "$(ENSEMBLE)" || (echo "Usage: make list-ensemble ENSEMBLE=submit-v1"; exit 1)
	@MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) uv run python3 scripts/promote.py list $(ENSEMBLE)

clear-ensemble:
	@test -n "$(ENSEMBLE)" || (echo "Usage: make clear-ensemble ENSEMBLE=submit-v1"; exit 1)
	@MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) uv run python3 scripts/promote.py clear $(ENSEMBLE)

search-runs:
	MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) uv run python3 scripts/promote.py search $(if $(ALL),--all,$(if $(EXPERIMENT),--experiment $(EXPERIMENT),--all)) $(if $(FOLDS),--folds $(FOLDS)) $(if $(MODELS),--model $(MODELS)) $(if $(SEEDS),--seeds $(SEEDS)) $(if $(FEATURES),--features $(FEATURES))
