"""The pins in pyproject.toml and the requirements files must agree."""
import os
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _lines(name):
    with open(os.path.join(ROOT, name)) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith(("#", "-r"))]


def test_pyproject_pins_match_requirements():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        project = tomllib.load(f)["project"]
    assert project["dependencies"] == _lines("requirements.txt")
    assert project["optional-dependencies"]["dev"] == _lines("requirements-dev.txt")


def test_python_version_pin_is_supported():
    with open(os.path.join(ROOT, ".python-version")) as f:
        major, minor = map(int, f.read().strip().split(".")[:2])
    assert (major, minor) >= (3, 12)
