install:
	uv sync
	uv run pre-commit install

remove-environment:
	rm -rf .venv

re-install: remove-environment
	$(MAKE) install

download-competition-data:
	if [ -z "$(KAGGLE_COMPETITION)" ]; then echo "KAGGLE_COMPETITION is unset" && exit 1; fi
	mkdir -p data/$${KAGGLE_COMPETITION%%-*}
	uv tool run kaggle competitions download -c ${KAGGLE_COMPETITION} -p data/$${KAGGLE_COMPETITION%%-*}/
	unzip -o data/$${KAGGLE_COMPETITION%%-*}/${KAGGLE_COMPETITION}.zip -d data/$${KAGGLE_COMPETITION%%-*}/${KAGGLE_COMPETITION}
	@if [ ! -d "competitions/${KAGGLE_COMPETITION}" ]; then \
		echo "Scaffolding competitions/${KAGGLE_COMPETITION}..."; \
		mkdir -p competitions/${KAGGLE_COMPETITION}; \
		uv init --name ${KAGGLE_COMPETITION} --no-package --directory competitions/${KAGGLE_COMPETITION}; \
		python3 -c "import pathlib; p = pathlib.Path('competitions/${KAGGLE_COMPETITION}/pyproject.toml'); t = p.read_text(); t = t.replace('dependencies = []', 'dependencies = [\n    \"kego\",\n]'); t += '\n[tool.uv.sources]\nkego = { workspace = true }\n'; p.write_text(t)"; \
		python3 -c "import re, pathlib; entry = 'competitions/${KAGGLE_COMPETITION}'; p = pathlib.Path('pyproject.toml'); t = p.read_text(); t = re.sub(r'(members = \[)', r'\1\n    \"' + entry + '\",', t) if entry not in t else t; p.write_text(t)"; \
		echo "Done. Add extra deps with: uv add --directory competitions/${KAGGLE_COMPETITION} <pkg>"; \
	else \
		echo "competitions/${KAGGLE_COMPETITION} already exists, skipping scaffold."; \
	fi

setup-new-competition:
	if [ -z ${KAGGLE_COMPETITION} ]; then echo "KAGGLE_COMPETITION is unset" && exit 1; else echo "KAGGLE_COMPETITION is set to '${KAGGLE_COMPETITION}'"; fi
	$(eval path_current := ${PWD})
	$(eval path_competition := "competitions/${KAGGLE_COMPETITION}")
	mkdir ${path_competition}
	uv init --name ${KAGGLE_COMPETITION} --no-package --directory ${path_competition}


fleet-register:
	uv run python -c "from kego.fleet import detect_machine, register_self, registration_summary; m = detect_machine(); print(registration_summary(m, register_self('fleet.toml', m)))"

test:
	uv run pytest tests/ -v

publish:
	rm -rf dist
	uv build
	uv publish
