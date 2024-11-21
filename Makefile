install-all:
	poetry install --all-extras

re-install-all:
	rm -rf .venv
	$(MAKE) install-all

install-pii:
	poetry install -E "torch-cuda121 pii"

install-czii:
	poetry install -E "czii"

ariel-build-features:
	poetry run python mlframework/ariel/features.py

download-competition-data:
	echo ${KAGGLE_COMPETITION}
	KAGGLE_COMPETITION='$(KAGGLE_COMPETITION)'
	mkdir -p data/$${KAGGLE_COMPETITION%%-*}
	poetry run kaggle competitions download -c ${KAGGLE_COMPETITION} -p data/$${KAGGLE_COMPETITION%%-*}/
	unzip data/$${KAGGLE_COMPETITION%%-*}/${KAGGLE_COMPETITION}.zip -d data/$${KAGGLE_COMPETITION%%-*}/${KAGGLE_COMPETITION}
