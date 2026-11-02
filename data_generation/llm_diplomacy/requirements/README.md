# Requirements files

- `upstream_requirements_archive.txt`: package list from the **uploaded archived copy** of `AI_Diplomacy/requirements.txt`, retained as provenance/reference, not copied from a current live dependency lock.
- `vllm_server_optional.txt`: model server dependency reference. Install vLLM in an environment compatible with your GPU drivers and Python version, following its own installation documentation; do not assume that the archived upstream app's Python 3.13 environment can serve as the vLLM environment unchanged.
- The helper scripts in this package use the Python standard library and need no pip dependencies by themselves.
- The source archive's `pyproject.toml` requires Python >= 3.13 and upstream recommends `uv sync` for the application. It also contained `uv.lock`, which is not redistributed here. For exact reproduction, retain the authorised upstream revision and its lockfile.
