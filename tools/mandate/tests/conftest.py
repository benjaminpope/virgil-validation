import csv
import pathlib
import shutil
import subprocess
import sys

import pytest
import yaml

EXAMPLES = pathlib.Path(__file__).resolve().parents[1] / "examples"

# A step that writes result.json from its order: status and checks come from the unit's columns.
STEP = (
    f"{sys.executable} -c \"import json,os; o=json.load(open(os.environ['MANDATE_ORDER'])); u=o['units'][0]; "
    "json.dump({'status': u.get('want','pass') or 'pass', 'checks': {'ok': u.get('want','pass') or 'pass'}, "
    "'n_inputs': sum(len(v) for v in o['inputs'].values())}, open(os.path.join(o['out'],'result.json'),'w'))\""
)


def commit(root: pathlib.Path, message: str = "x") -> None:
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.org", "commit", "-qm", message],
        check=True,
    )


@pytest.fixture
def make(tmp_path, monkeypatch):
    """Build a committed campaign from a plan dict and unit rows."""
    monkeypatch.setenv("MANDATE_ACTOR", "agent-test")
    monkeypatch.delenv("MANDATE_RUNS", raising=False)

    def build(plan: dict, rows: list[dict]) -> pathlib.Path:
        root = tmp_path / "campaign"
        (root / "registry").mkdir(parents=True)
        (root / "mandate.yaml").write_text(yaml.safe_dump({"humans": ["ben"], "unit_id": "id", **plan}))
        with open(root / "registry" / "units.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (root / ".gitignore").write_text("runs/\n")
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        commit(root)
        return root

    return build


@pytest.fixture
def demo(tmp_path, monkeypatch):
    """The NIRCam demo campaign, copied into a fresh git repository."""
    pytest.importorskip("numpy")
    monkeypatch.setenv("MANDATE_ACTOR", "agent-test")
    monkeypatch.delenv("MANDATE_RUNS", raising=False)
    root = tmp_path / "demo"
    shutil.copytree(EXAMPLES / "nircam_demo", root, ignore=shutil.ignore_patterns("demo.sh", "runs", "ledger"))
    text = (root / "mandate.yaml").read_text().replace("python steps/toy.py", f"{sys.executable} steps/toy.py")
    (root / "mandate.yaml").write_text(text)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    commit(root)
    return root
