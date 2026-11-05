"""Execute and save the Assessment 1 notebook using the current Python environment."""
from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.pop("PYTHONSTARTUP", None)
for variable, folder in {
    "MPLCONFIGDIR": "matplotlib", "IPYTHONDIR": "ipython",
    "JUPYTER_CONFIG_DIR": "jupyter-config", "JUPYTER_DATA_DIR": "jupyter-data",
    "JUPYTER_RUNTIME_DIR": "jupyter-runtime",
}.items():
    path = ROOT / "tmp" / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(path))

import nbformat
from nbclient import NotebookClient

path = ROOT / "notebooks/01_assessment1_eda.ipynb"
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
started = time.monotonic()


def progress(cell, cell_index):
    if cell.cell_type == "code":
        print(f"Executing code cell at notebook position {cell_index+1}/{len(notebook.cells)}", flush=True)


client = NotebookClient(
    notebook, timeout=300, kernel_name="python3",
    resources={"metadata": {"path": str(ROOT / "notebooks")}},
    on_cell_start=progress,
)
client.execute()
nbformat.write(notebook, path)
print(f"Completed Run All in {time.monotonic()-started:.1f}s: {path.relative_to(ROOT)}", flush=True)
