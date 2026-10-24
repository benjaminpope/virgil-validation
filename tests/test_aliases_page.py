"""The period-alias pages, one per system, and the pull-back that files a run by its own record."""

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import build_aliases_page as B  # noqa: E402
import pull_back_orbit_results as P  # noqa: E402

pytestmark = pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")


def test_cold_is_read_from_the_run_record():
    assert B.is_cold({"args": {"no_init": True}, "init": None})
    assert not B.is_cold({"args": {"no_init": True}, "init": {"period": 12.1}})
    assert not B.is_cold({"args": {"no_init": False}})
    assert not B.is_cold({})  # no record: not known to be cold


def test_each_system_has_its_own_page():
    pages = B.render()
    assert set(pages) == set(B._registry())
    gl = pages["gl229"]
    assert gl.startswith("# Period aliases: Gl 229 Ba-Bb\n") and "| N | P (d) |" in gl and "(../index.md)" in gl
    assert "../../assets/orbit_aliases/gl229/sky.png" in gl
    assert "$" + "SYSTEM" not in gl and "Release gate" not in gl


def test_gallery_tiles_of_each_system():
    assert B.tiles("gl229") == ("gl229b", "gl229b-blind") and B.tiles("omi_leo") == ("omi-leo",)


def test_pull_back_files_a_run_by_its_own_record(tmp_path):
    def run(name, no_init, init=None):
        d = tmp_path / name / "gl229"
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"args": {"no_init": no_init}, "init": init}))
        return d
    assert P.destination(run("tag-cold", False, {"period": 12.1})) == "gl229"  # warm, though the tag says cold
    assert P.destination(run("tag-warm", True)) == "gl229-cold"  # cold, though the tag does not say so
    nometa = tmp_path / "x" / "gl229"
    nometa.mkdir(parents=True)
    assert P.destination(nometa) is None
