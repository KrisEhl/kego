CLUSTER_HOST ?= kristian@omarchyd
CLUSTER_PLAYGROUND ?= ~/projects/kego/competitions/playground

# Cluster management (run from local Mac)
cluster-start:
	ssh $(CLUSTER_HOST) "bash -lc 'cd $(CLUSTER_PLAYGROUND) && git pull --ff-only && uv run make start-head && uv run make mlflow-start'"

cluster-stop:
	ssh $(CLUSTER_HOST) "bash -lc 'cd $(CLUSTER_PLAYGROUND) && uv run make stop && uv run make mlflow-stop'"

cluster-status:
	ssh $(CLUSTER_HOST) "bash -lc 'cd $(CLUSTER_PLAYGROUND) && uv run ray status'"

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
	$(eval path_competition := "competitions/${KAGGLE_COMPETITION}")
	mkdir ${path_competition}
	uv init --name ${KAGGLE_COMPETITION} --no-package --directory ${path_competition}


test:
	uv run pytest tests/ -v

publish:
	rm -rf dist
	uv build
	uv publish
