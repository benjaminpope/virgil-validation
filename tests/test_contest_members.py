"""The contest runner's evidence grid and ensemble members (scripts/contest_images.py).

``grow_grid`` must extend the (ℓ, σ) grid by factors of 2 while the best point
sits on an edge that can move, and stop at the limits; ``member_settings`` must
cycle the central source through point / none / disk / none and give halos
only to the tasks that ask for them. Both are pure Python: no fits are run.
"""

import importlib.util
import pathlib

import pytest

SPEC = importlib.util.spec_from_file_location(
    "contest_images", pathlib.Path(__file__).parents[1] / "scripts" / "contest_images.py"
)
contest_images = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contest_images)


def _peak(at_length, at_sigma):
    """A log evidence peaked (in log space) at (at_length, at_sigma)."""
    import math

    def evaluate(length, sigma):
        return -(math.log2(length / at_length) ** 2) - math.log2(sigma / at_sigma) ** 2

    return evaluate


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_grid_grows_towards_an_edge_optimum_and_stops_there():
    ls, ss, values = contest_images.grow_grid(_peak(0.25, 16.0), [0.5, 1.0, 2.0], [1.0, 2.0, 4.0],
                                             (0.125, 8.0), (0.25, 16.0), growths=5)
    # One row beyond the optimum confirms it is interior; σ stops at its limit.
    assert ls == [0.125, 0.25, 0.5, 1.0, 2.0]
    assert ss == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert max(values, key=values.get) == (0.25, 16.0)
    assert len(values) == len(ls) * len(ss)  # the grid stays rectangular


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_grid_does_not_grow_for_an_interior_optimum_or_past_its_limits():
    ls, ss, values = contest_images.grow_grid(_peak(1.0, 2.0), [0.5, 1.0, 2.0], [1.0, 2.0, 4.0],
                                             (0.125, 8.0), (0.25, 16.0), growths=3)
    assert (ls, ss, len(values)) == ([0.5, 1.0, 2.0], [1.0, 2.0, 4.0], 9)
    # The optimum lies beyond the limits: growth stops at them.
    ls, ss, _ = contest_images.grow_grid(_peak(0.01, 100.0), [0.5, 1.0, 2.0], [1.0, 2.0, 4.0],
                                         (0.125, 8.0), (0.25, 16.0), growths=10)
    assert ls[0] == 0.125 and ss[-1] == 16.0


@pytest.mark.validates("pipeline:contest-imaging", roots=["mathematics"], kind="reference")
def test_members_cycle_the_central_source_and_halo_only_where_asked():
    labels = [spec["label"] for spec in contest_images.TASKS]
    data2, data1 = labels.index("2004_data2"), labels.index("2004_data1")
    models = [contest_images.member_settings(data2, m)["star_model"] for m in range(8)]
    assert models == ["point", "none", "disk", "none"] * 2
    assert [contest_images.member_settings(data2, m)["halo"] for m in range(8)] == [False] * 4 + [True] * 4
    assert not any(contest_images.member_settings(data1, m)["halo"] for m in range(8))
    assert all(contest_images.member_settings(data2, m)["star"] == (models[m] != "none") for m in range(8))


@pytest.mark.validates("pipeline:contest-imaging", "virgil.grid_fit.linear_flux_grid", roots=["self-consistency"], kind="reference")
@pytest.mark.skipif(not (pathlib.Path("~/data/imaging_contests/2004/2004-data2.fits").expanduser().exists()),
                    reason="contest data not fetched")
def test_companion_search_finds_a_simulated_companion():
    import jax
    import virgil.models as vm
    from virgil.imaging import beam, starting_image
    from virgil.oidata import OIData

    template = OIData(str(pathlib.Path("~/data/imaging_contests/2004/2004-data2.fits").expanduser()))
    truth = vm.System(star=vm.PointSource(), comp=vm.PointSource(0.1, dra=6.0, ddec=-3.0))
    data = template.with_model(truth, key=jax.random.PRNGKey(0))
    resolution = beam(data)
    start = starting_image(data, star=True)
    base, priors = contest_images.companion_search(data, resolution, vm.PointSource(), {}, start)
    assert isinstance(base, vm.System)
    assert float(base.comp.dra) == pytest.approx(6.0, abs=0.3)
    assert float(base.comp.ddec) == pytest.approx(-3.0, abs=0.3)
    assert float(base.comp.flux) == pytest.approx(0.1, rel=0.2)
    assert {"star.comp.dra", "star.comp.ddec", "star.comp.flux"} <= set(priors)
