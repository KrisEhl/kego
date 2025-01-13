install-all:
	poetry install --all-extras

remove-environment:
	rm -rf .venv

re-install-all: remove-environment
	$(MAKE) install-all

install-pii:
	poetry install -E "torch pii"

install-torch:
	poetry install -E "torch pii"

install-czii-monai:
	poetry install -E "torch czii monai"

install-czii:
	poetry install -E "czii"

ariel-build-features:
	poetry run python kego/ariel/features.py

download-competition-data:
	echo ${KAGGLE_COMPETITION}
	KAGGLE_COMPETITION='$(KAGGLE_COMPETITION)'
	mkdir -p data/$${KAGGLE_COMPETITION%%-*}
	poetry run kaggle competitions download -c ${KAGGLE_COMPETITION} -p data/$${KAGGLE_COMPETITION%%-*}/
	unzip data/$${KAGGLE_COMPETITION%%-*}/${KAGGLE_COMPETITION}.zip -d data/$${KAGGLE_COMPETITION%%-*}/${KAGGLE_COMPETITION}
