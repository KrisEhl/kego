install-all:
	poetry install --all-extras

re-install-all:
	rm -rf .venv
	$(MAKE) install-all

install-pii:
	poetry install -E "torch-cuda121 pii"

ariel-build-features:
	poetry run python mlframework/ariel/features.py
