"""The failure modes of OzSTAR job 18296568 in scripts/reanalyse_orbits.py: list-valued scales, grid-edge positions."""

import pathlib
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import reanalyse_orbits as R  # noqa: E402

pytest.importorskip("virgil.epochs")
# How the script behaves, not a check of virgil against an independent result: guards.
pytestmark = pytest.mark.validates("evidence", roots=["standards"], kind="guard")


class _Result:
    def table(self):
        row = dict(
            n=3,
            period=12.0,
            chi2_red=1.2,
            log_z=-5.0,
            log_z_is=-5.1,
            p=0.9,
            ess=300.0,
            flagged=False,
            flags=[],
        )
        # per-block scales as lists (old virgil), flat floats (new), and None / inf entries
        return [
            dict(row, scales=[[1.0, 2.0], [1.5]]),
            dict(row, scales=[1.0, None, np.inf]),
            dict(row, scales=[], flags=None),
        ]


def test_print_table_accepts_nested_and_flat_scales(capsys):
    R.print_table(_Result())
    out = capsys.readouterr().out
    assert out.count("\n") == 4 and "2.00" in out


def test_positions_for_widens_until_no_epoch_is_at_the_edge(monkeypatch):
    import virgil.epochs as E

    halves = []

    class Ep:
        def __init__(self, edge):
            self.edge = np.array(edge)

        def positions(self, t_ref):
            return ("positions", t_ref)

    def fake_epoch_positions(epochs, grid, n_peaks):
        halves.append(float(grid["dra"].max()))
        return Ep([len(halves) < 3, False])  # at the edge for the first two grids

    monkeypatch.setattr(E, "epoch_positions", fake_epoch_positions)
    monkeypatch.setattr(E, "Epochs", lambda d: d)
    d = types.SimpleNamespace(
        wavel=np.array([2.2e-6]), u=np.array([100.0]), v=np.array([0.0])
    )
    loaded = R.Loaded(["a", "b"], [d, d], [d, d], 10.0, (1.0, 5.0))
    pos, ep, info = R.positions_for(loaded, 60000.0)
    assert len(halves) == 3 and halves[2] > halves[1] > halves[0]
    assert (
        info["edge_epochs"] == []
        and len(info["half_widths"]) == 3
        and pos[0] == "positions"
    )


def test_positions_for_records_a_persistent_edge_and_continues(monkeypatch):
    import virgil.epochs as E

    class Ep:
        edge = np.array([False, True])

        def positions(self, t_ref):
            return "p"

    monkeypatch.setattr(E, "epoch_positions", lambda epochs, grid, n_peaks: Ep())
    monkeypatch.setattr(E, "Epochs", lambda d: d)
    d = types.SimpleNamespace(
        wavel=np.array([2.2e-6]), u=np.array([100.0]), v=np.array([0.0])
    )
    loaded = R.Loaded(["a", "b"], [d, d], [d, d], 10.0, (1.0, 5.0))
    _, _, info = R.positions_for(loaded, 60000.0, max_widen=2)
    assert info["edge_epochs"] == ["b"] and len(info["half_widths"]) == 3


def test_check_resume_keeps_matching_bands_and_moves_mismatched_aside(tmp_path):
    fp = dict(system="x", args=dict(n_is=10))
    (tmp_path / "bands").mkdir()
    R._check_resume(tmp_path, fp)  # no fingerprint yet: stale, moved aside
    assert (
        not (tmp_path / "bands").exists()
        and len(list(tmp_path.glob("bands.stale-*"))) == 1
    )
    (tmp_path / "bands").mkdir()
    R._check_resume(tmp_path, fp)  # same fingerprint: kept
    assert (tmp_path / "bands").exists()
    R._check_resume(tmp_path, dict(fp, args=dict(n_is=11)))  # changed setting
    assert not (tmp_path / "bands").exists()


def test_plot_corner_reports_a_collapsed_sample_instead_of_crashing_in_corner(tmp_path):
    pytest.importorskip("corner")
    keys = ["period", "dt_peri", "ecc", "inc", "omega", "Omega", "a_mas", "flux"]
    samples = {
        k: np.full(50, 1.0) for k in keys
    }  # hd70937 in job 18310053: every draw the same
    result = types.SimpleNamespace(
        samples={3: samples}, best=types.SimpleNamespace(n=3)
    )
    with pytest.raises(ValueError, match="collapsed"):
        R.plot_corner(
            tmp_path / "c.png", result
        )  # caught and recorded by run()'s attempt()
