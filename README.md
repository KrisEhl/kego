## kego

## Setup
To make VScode see poetry kernels use:
```bash
poetry config virtualenvs.in-project true
```
Make sure notebook use newest version of code by adding
```python
%load_ext autoreload
%autoreload 2
```
## Competition specific
### Ariel
- `.env`
```bash
FOLDER_COMPETITION=${PATH_EFOLDER}/ariel-data-challenge-2024/
```
