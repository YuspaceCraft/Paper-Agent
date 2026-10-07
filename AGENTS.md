# Agent Notes

## Python Environment

The canonical project environment is the conda environment `demo`.

Before running any Python command, test, package install, or backend command:

```bash
conda activate demo
```

After activation, use:

```bash
python -m pytest ...
python -m compileall ...
python -m web.cli ...
uvicorn web.api.main:app ...
```

Do not use the Codex process interpreter or another base Python environment to
conclude that project dependencies are missing. Verify the active environment
first.

For a non-interactive shell that cannot load conda activation, use:

```bash
conda run -n demo python ...
```

The direct interpreter path is a last-resort fallback:

```text
C:/Users/30811/miniconda3/envs/demo/python.exe
```

## Frontend

```bash
cd web/frontend
npm run build:renderer
npm run lint
```
