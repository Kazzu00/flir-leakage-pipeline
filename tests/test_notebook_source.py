import json
import runpy
from pathlib import Path

import pytest


def test_notebook_check_distinguishes_narrative_from_private_state(tmp_path: Path) -> None:
    check = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_notebook_source.py"))["check_notebook_source"]
    path = tmp_path / "source.ipynb"
    notebook = {"cells": [{"cell_type": "markdown", "source": ["Context:\n", "Narrative"]}], "metadata": {}}
    path.write_text(json.dumps(notebook))
    check(path)
    notebook["cells"][0]["source"] = ["C:/private/dataset"]
    path.write_text(json.dumps(notebook))
    with pytest.raises(ValueError, match="absolute path"):
        check(path)
    notebook["cells"][0] = {"cell_type": "code", "source": ["print(1)"], "outputs": [], "execution_count": 0}
    path.write_text(json.dumps(notebook))
    with pytest.raises(ValueError, match="execution state"):
        check(path)
