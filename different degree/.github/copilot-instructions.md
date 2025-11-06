## Purpose
This file gives concise, actionable context for an AI coding agent working in this repository. The project is currently a tiny single-script Python project that includes a local virtual environment (`myvirtual/`) and one script (`new.py`).

## Big picture
- Single-script utility: `new.py` is the main (and only) runnable file. It uses NumPy from the repository-local virtualenv at `myvirtual/Lib/site-packages/numpy`.
- No services, no CI, no packages or modules yet — treat this as a small, local script project. If you add packages or a package layout, update this file.

## Important local files / layout
- `new.py` — the script. Example: it imports numpy and runs `a = np.array([1,2,3])` then prints `a`.
- `myvirtual/` — a checked-in virtual environment. It contains `Scripts/` (Windows activators) and `Lib/site-packages/` with `numpy/` already installed.

## How to run (Windows PowerShell)
Use the repository's virtualenv to run or debug. In PowerShell:

```powershell
# activate the venv
.\myvirtual\Scripts\Activate.ps1
# run the script with the venv python
python .\new.py
```

Alternative (without activating):

```powershell
.\myvirtual\Scripts\python.exe .\new.py
```

Note: `new.py` includes a shebang line that references the venv python; if you move or recreate the venv, update or remove that line so the active interpreter is used.

## Dependency and environment notes
- The project currently pins no explicit requirements file. To capture installed packages, run inside the activated venv:

```powershell
pip freeze > requirements.txt
```

- The presence of `myvirtual/` in the repo means dependencies are bundled locally. Do not modify files inside `myvirtual/` unless you are intentionally changing the environment. Prefer creating/updating `requirements.txt` instead of editing `myvirtual/` contents by hand.

## Conventions & patterns in this repo
- Single-file scripts go in the repo root (e.g., `new.py`).
- When adding Python modules, create a top-level package directory (e.g., `src/` or package name`) and add tests alongside under a `tests/` folder.
- Avoid committing environment-specific binaries. If you introduce a proper gitignore, ensure `myvirtual/` is intentionally tracked or listed there.

## What to watch for when editing
- Keep imports explicit (e.g., `import numpy as np`) — this is the existing style in `new.py`.
- `new.py` uses simple procedural code; if you refactor into functions or modules, add a small entrypoint (`if __name__ == "__main__":`) to preserve script behavior.

## Missing/absent features (observed)
- No test framework, no CI workflow, no README. AI agents should not assume tests or build steps exist — add them explicitly if implementing features that require verification.

## When you make changes
- If you add dependencies, update or create `requirements.txt` from the activated venv.
- If you create new Python modules, add a short note in `README.md` describing how to run and where the entrypoint is.

## Quick examples from the codebase
- `new.py` (root):

```python
import numpy as np
a = np.array([1,2,3])
print(a)
```

Use this example when suggesting API changes or adding small features that use numpy.

## Questions for the repo owner
- Should `myvirtual/` remain in version control or should it be added to `.gitignore` and replaced by a `requirements.txt` + venv creation instructions?
- Do you want a preferred package layout (e.g., `src/` + `tests/`) if this project grows?

If anything above is unclear or you want a different tone/level of guidance, tell me what to change and I'll iterate.
