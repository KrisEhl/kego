install-all:
	poetry install --all-extras

install-pii:
	poetry install -E "torch-cuda121 pii"

ariel-build-features:
	poetry run python mlframework/ariel/features.py
