## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python environment and dependency management. All required package versions are pinned in [`pyproject.toml`](pyproject.toml) and locked in [`uv.lock`](uv.lock) to ensure a fully reproducible environment.

### 1. Install `uv`

If you don't already have `uv` installed, follow the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/), e.g.:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create the environment

From the root of the repository, simply run:

```bash
uv sync
```

This reads `pyproject.toml` and `uv.lock` and creates a local `.venv` with the exact package versions used to produce the paper's results (Python ≥ 3.11 is required, see [`.python-version`](.python-version)).

Any script in this repository can then be run inside the managed environment with:

```bash
uv run <script.py>
```