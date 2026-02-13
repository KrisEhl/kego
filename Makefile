install:
	uv sync
	uv run pre-commit install

remove-environment:
	rm -rf .venv

re-install: remove-environment
	$(MAKE) install

download-competition-data:
	echo ${KAGGLE_COMPETITION}
	KAGGLE_COMPETITION='$(KAGGLE_COMPETITION)'
	mkdir -p data/$${KAGGLE_COMPETITION%%-*}
	uv tool run kaggle competitions download -c ${KAGGLE_COMPETITION} -p data/$${KAGGLE_COMPETITION%%-*}/
	unzip data/$${KAGGLE_COMPETITION%%-*}/${KAGGLE_COMPETITION}.zip -d data/$${KAGGLE_COMPETITION%%-*}/${KAGGLE_COMPETITION}

setup-new-competition:
	if [ -z ${KAGGLE_COMPETITION} ]; then echo "KAGGLE_COMPETITION is unset" && exit 1; else echo "KAGGLE_COMPETITION is set to '${KAGGLE_COMPETITION}'"; fi
	$(eval path_current := ${PWD})
	$(eval path_competition := "notebooks/${KAGGLE_COMPETITION}")
	mkdir ${path_competition}
	uv init --name ${KAGGLE_COMPETITION} --no-package --directory ${path_competition}


publish:
	rm -rf dist
	uv build
	uv publish

register-head:
	cd cluster && RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 uv run ray start --head --port=${RAY_API_SERVER_PORT} --node-ip-address ${RAY_API_SERVER_IP} --dashboard-host=0.0.0.0 --dashboard-port=8265 --ray-client-server-port=10001 --num-cpus=$$(expr $$(nproc --all) - 2)

register-worker:
	$(eval NODE_IP := $(shell if grep -qi microsoft /proc/version 2>/dev/null; then ip -4 addr show | grep -oP 'inet 192\.168\.\d+\.\d+' | head -1 | grep -oP '192\.168\.\d+\.\d+'; fi))
	cd cluster && RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1 uv run ray start --address="${RAY_API_SERVER_IP}:${RAY_API_SERVER_PORT}" $(if $(NODE_IP),--node-ip-address=$(NODE_IP))

restart-register-worker:
	cd cluster && uv run ray stop --force
	$(MAKE) register-worker
