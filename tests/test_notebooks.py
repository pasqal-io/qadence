"""Test examples scripts."""
from __future__ import annotations

import os
import subprocess
import sys
from glob import glob
from pathlib import Path
from typing import List

import pytest

expected_fail = {}  # type: ignore


def get_ipynb_files(dir: Path) -> List[Path]:
    files = []

    for it in dir.iterdir():
        if it.suffix == ".ipynb":
            files.append(it)
        elif it.is_dir():
            files.extend(get_ipynb_files(it))
    return files


# FIXME: refactor choice of notebooks folders
docsdir = Path(__file__).parent.parent.joinpath("docs")
notebooks = [
    Path(nb).relative_to(docsdir.parent) for nb in glob(str(docsdir / "**/*.ipynb"), recursive=True)
]
for example, reason in expected_fail.items():
    try:
        notebooks[notebooks.index(Path(example))] = pytest.param(  # type: ignore
            example, marks=pytest.mark.xfail(reason=reason)
        )
    except ValueError:
        pass


@pytest.mark.parametrize("notebook", notebooks, ids=map(str, notebooks))
def test_notebooks(notebook: Path) -> None:
    """Execute docs notebooks as a test, passes if it returns 0."""
    jupyter_cmd = ["-m", "jupyter", "nbconvert", "--to", "python", "--execute"]
    path = str(notebook)
    cmd = [sys.executable, *jupyter_cmd, path]
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ}  # type: ignore
    ) as run_example:
        stdout, stderr = run_example.communicate()
        error_string = (
            f"Notebook {path} failed\n" f"stdout:{stdout.decode()}\n" f"stderr: {stderr.decode()}"
        )

    if run_example.returncode != 0:
        raise Exception(error_string)
