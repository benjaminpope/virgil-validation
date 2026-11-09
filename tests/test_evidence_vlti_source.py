"""The run header records which virgil-vlti commit the evidence ran against (PEP 610)."""

import json
from importlib import metadata

import pytest

from evidence import meta

SHA = "29dfaa77dd80338b9cf25d9c354e4055c6f93c4a"


class _Dist:
    def __init__(self, text):
        self.text = text

    def read_text(self, name):
        assert name == "direct_url.json"
        return self.text


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_git_source_reads_the_installers_record(monkeypatch):
    record = {"url": "https://github.com/benjaminpope/virgil-vlti", "vcs_info": {"vcs": "git", "commit_id": SHA}}
    monkeypatch.setattr(metadata, "distribution", lambda name: _Dist(json.dumps(record)))
    assert meta._git_source("virgil-vlti") == {"commit": SHA, "url": record["url"]}


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_git_source_without_a_git_record_or_install(monkeypatch):
    monkeypatch.setattr(metadata, "distribution", lambda name: _Dist(None))
    assert meta._git_source("virgil-vlti") == {"commit": None, "url": None}

    def missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "distribution", missing)
    assert meta._git_source("virgil-vlti") == {"commit": None, "url": None}


@pytest.mark.validates("evidence", roots=["self-consistency"], kind="guard")
def test_run_header_names_virgil_vlti(monkeypatch):
    monkeypatch.setattr(meta, "_git_source", lambda name: {"commit": SHA, "url": "u"})
    header = meta.run_header()
    assert header["virgil_vlti"]["commit"] == SHA
    assert "virgil-vlti" in header["versions"]
