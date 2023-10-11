# CHANGELOG

Change log for the codebase. Initialised from the developments following version `V0.11.3`

## V0.12.0

    - Introduction of pre-commit hooks (Ruff & Blacks)
    - Linting of code
    - Creation of feature selection mode for AO
    - max_features kwarg added for feature selection
    - moved source code into an src folder & modified dockerfile
    - created utils subfolder
    - created models subfolder
    - added in structure.md within ./src to explain the structure format adopted
    - renamed mode scripts to have a `mode_` pre-fix
    - added extra assertion in mode checks
    - removed redundant `check_config`
    - refactored `plot_graphs` to remove unnessicary `plot_dict` var
