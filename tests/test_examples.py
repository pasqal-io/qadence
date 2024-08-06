"""Test examples scripts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pytest

expected_fail: dict = {"backends/low_level/profiling_debugging.py": "Requires CUDA device."}


def get_py_files(dir: Path) -> Iterable[Path]:
    files = []

    for it in dir.iterdir():
        if it.suffix == ".py":
            files.append(it)
        elif it.is_dir():
            files.extend(get_py_files(it))
    return files


examples_dir = Path(__file__).parent.parent.joinpath("examples").resolve()
assert examples_dir.exists()
examples = get_py_files(examples_dir)
example_names = [f"{example.relative_to(examples_dir)}" for example in examples]
for example, reason in expected_fail.items():
    try:
        examples[example_names.index(example)] = pytest.param(  # type: ignore
            example, marks=pytest.mark.xfail(reason=reason)
        )
    except ValueError:
        pass


@pytest.mark.parametrize("example", examples, ids=example_names)
def test_example(example: Path) -> None:
    """Execute and example as a test, passes if it returns 0."""
    cmd = [sys.executable, example]
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env={**os.environ}  # type: ignore
    ) as run_example:
        stdout, stderr = run_example.communicate()
        error_string = (
            f"Example {example.name} failed\n"
            f"stdout:{stdout.decode()}\n"
            f"stderr: {stderr.decode()}"
        )
    if run_example.returncode != 0:
        raise Exception(error_string)
