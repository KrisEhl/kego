RAY_HEAD_IP ?= 192.168.178.32
RAY_HEAD_PORT ?= 6379
RAY_DASHBOARD ?= http://$(RAY_HEAD_IP):8265
MLFLOW_PORT ?= 5000
MLFLOW_DIR ?= $(HOME)/mlflow
MLFLOW_TRACKING_URI ?= http://$(RAY_HEAD_IP):$(MLFLOW_PORT)

start-head:
	RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 uv run ray start --head --port=$(RAY_HEAD_PORT) --node-ip-address $(RAY_HEAD_IP) --dashboard-host=0.0.0.0 --dashboard-port=8265 --ray-client-server-port=10001 --num-cpus=$$(expr $$(nproc --all) - 2)

start-worker:
	$(eval NODE_IP := $(shell if grep -qi microsoft /proc/version 2>/dev/null; then ip -4 addr show | grep -oP 'inet 192\.168\.\d+\.\d+' | head -1 | grep -oP '192\.168\.\d+\.\d+'; fi))
	RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 uv run ray start --address="$(RAY_HEAD_IP):$(RAY_HEAD_PORT)" $(if $(NODE_IP),--node-ip-address=$(NODE_IP))

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
	@JOB_ID=$$(uv run ray job list --address $(RAY_DASHBOARD) 2>/dev/null | \
		python3 -c "import sys,re; text=sys.stdin.read(); \
		jobs=re.findall(r\"submission_id='(raysubmit_\w+)'.*?status=<JobStatus\.(\w+):\",text,re.DOTALL); \
		running=[j for j,s in jobs if s=='RUNNING']; \
		print(running[-1] if running else '')"); \
	if [ -z "$$JOB_ID" ]; then echo "No running jobs"; \
	else uv run ray job logs "$$JOB_ID" --address $(RAY_DASHBOARD) 2>/dev/null | \
		python3 ../scripts/train_logs.py "$$JOB_ID"; \
	fi

N ?= 20
logs-raw:
	@JOB_ID=$$(uv run ray job list --address $(RAY_DASHBOARD) 2>/dev/null | \
		python3 -c "import sys,re; text=sys.stdin.read(); \
		jobs=re.findall(r\"submission_id='(raysubmit_\w+)'.*?status=<JobStatus\.(\w+):\",text,re.DOTALL); \
		running=[j for j,s in jobs if s=='RUNNING']; \
		print(running[-1] if running else '')"); \
	if [ -z "$$JOB_ID" ]; then echo "No running jobs"; \
	else uv run ray job logs "$$JOB_ID" --address $(RAY_DASHBOARD) 2>/dev/null | tail -n $(N); \
	fi

mlflow-start:
	mkdir -p $(MLFLOW_DIR)/artifacts
	uv run mlflow server \
		--backend-store-uri sqlite:///$(MLFLOW_DIR)/mlflow.db \
		--default-artifact-root $(MLFLOW_DIR)/artifacts \
		--host 0.0.0.0 --port $(MLFLOW_PORT) &

mlflow-stop:
	pkill -f "mlflow server" || true

log-score:
	@test -n "$(SCORE)" || (echo "Usage: make log-score SCORE=0.953"; exit 1)
	@uv run python3 -c "\
	import mlflow; \
	mlflow.set_tracking_uri('$(MLFLOW_TRACKING_URI)'); \
	runs = mlflow.search_runs(search_all_experiments=True, filter_string=\"run_name LIKE 'ensemble_%'\", order_by=['start_time DESC'], max_results=1); \
	assert len(runs) > 0, 'No ensemble runs found'; \
	rid = runs.iloc[0].run_id; \
	name = runs.iloc[0].run_name; \
	exp = mlflow.get_experiment(runs.iloc[0].experiment_id).name; \
	client = mlflow.tracking.MlflowClient(); \
	client.log_metric(rid, 'public_lb_score', $(SCORE)); \
	print(f'Logged public_lb_score=$(SCORE) to run \"{name}\" in experiment \"{exp}\"')"

submit-fast:
	uv run ray job submit --address $(RAY_DASHBOARD) --no-wait -- bash -c 'export KEGO_PATH_DATA=/home/kristian/projects/kego/data && export PYTHONUNBUFFERED=1 && export MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) && python ../notebooks/playground/train_s6e2_baseline.py --fast'

submit-full:
	uv run ray job submit --address $(RAY_DASHBOARD) --no-wait -- bash -c 'export KEGO_PATH_DATA=/home/kristian/projects/kego/data && export PYTHONUNBUFFERED=1 && export MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) && python ../notebooks/playground/train_s6e2_baseline.py'

submit-neural:
	uv run ray job submit --address $(RAY_DASHBOARD) --no-wait -- bash -c 'export KEGO_PATH_DATA=/home/kristian/projects/kego/data && export PYTHONUNBUFFERED=1 && export MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) && python ../notebooks/playground/train_s6e2_baseline.py --neural'

submit-debug:
	uv run ray job submit --address $(RAY_DASHBOARD) --no-wait -- bash -c 'export KEGO_PATH_DATA=/home/kristian/projects/kego/data && export PYTHONUNBUFFERED=1 && export MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) && python ../notebooks/playground/train_s6e2_baseline.py --debug --fast'
