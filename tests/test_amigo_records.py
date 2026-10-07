"""virgil.amigo.load_oi_data on a synthetic AMIGO mixed-DISCO product.

virgil documents the product (virgil.amigo, docs/amigo_disco.md and
docs/conventions.md): a ``.npy`` file holding a pickled ``{filter: record}``
dictionary; each record has ``u``, ``v`` (m), ``wavelength_m``,
``disco_coefficients`` and ``disco_sigma`` (independent errors, modes in
order of increasing error), and the operators ``disco_logamp_model_operator``
and ``disco_phase_model_operator`` (modes x uv points). The model vector of
a complex visibility V is ``A_logamp @ log|V| + A_phase @ arg V``, and the
product stores (u, v) with the opposite sign of the visibility convention,
which the reader negates.

We build such a product here from those statements alone: random
operators, a faint binary's closed-form visibility at the physical
(u, v) = -(stored u, v), and its DISCO coefficients. virgil must read back
our coefficients and errors, and its model of any binary must equal ours.
"""

import numpy as np
import pytest

from evidence.plugin import record

vm = pytest.importorskip("virgil.models")
amigo = pytest.importorskip("virgil.amigo")
from virgil.likelihood import whitened_residuals  # noqa: E402

pytestmark = pytest.mark.x64

MAS = np.pi / 180 / 3600 / 1000
TRUTH = (-34.7, 197.0, 1e-3)  # dra, ddec (mas), flux ratio
WAVELENGTH = 4.3e-6


def binary_vis(u, v, wl, dra, ddec, f):
    """Point primary at the origin, companion at (dra, ddec) mas East and
    North, sign exp(-2 pi i (u dra + v ddec) / lambda)."""
    return (1 + f * np.exp(-2j * np.pi * (u * dra + v * ddec) / wl * MAS)) / (1 + f)


def disco(rec, dra, ddec, f):
    vis = binary_vis(-rec["u"], -rec["v"], rec["wavelength_m"], dra, ddec, f)
    return rec["disco_logamp_model_operator"] @ np.log(np.abs(vis)) + rec["disco_phase_model_operator"] @ np.angle(vis)


def make_record(seed, n_uv=40, n_modes=60):
    rng = np.random.default_rng(seed)
    rec = {
        "u": rng.uniform(-6.0, 6.0, n_uv),
        "v": rng.uniform(-6.0, 6.0, n_uv),
        "wavelength_m": WAVELENGTH,
        "disco_sigma": np.sort(rng.uniform(1e-4, 1e-3, n_modes)),
        "disco_logamp_model_operator": rng.normal(size=(n_modes, n_uv)) / np.sqrt(n_uv),
        "disco_phase_model_operator": rng.normal(size=(n_modes, n_uv)) / np.sqrt(n_uv),
    }
    rec["disco_coefficients"] = disco(rec, *TRUTH)
    return rec


@pytest.fixture(scope="module")
def product(tmp_path_factory):
    records = {"F380M": make_record(1), "F430M": make_record(2)}
    path = tmp_path_factory.mktemp("amigo") / "calibrated_visibility.npy"
    np.save(path, records, allow_pickle=True)
    return path, records


def model_vector(data, scene):
    return np.asarray(data.model(scene))


@pytest.mark.validates("virgil.amigo.load_oi_data", roots=["mathematics"])
def test_reads_coefficients_and_models_them_as_documented(product):
    """Every filter: the data and errors are our coefficients and sigmas
    exactly; virgil's model of the true binary and of three others equals
    our closed-form DISCO vector (abs. 1e-12, the coefficients are of
    order 1e-3); the true binary leaves zero whitened residuals."""
    path, records = product
    loaded = amigo.load_oi_data(path)
    assert set(loaded) == set(records)
    worst = 0.0
    for name, rec in records.items():
        data = loaded[name]
        values, errors = (np.asarray(a) for a in data.flatten_data())
        np.testing.assert_array_equal(values, rec["disco_coefficients"])
        np.testing.assert_array_equal(errors, rec["disco_sigma"])
        for p in [TRUTH, (50.0, -80.0, 3e-3), (-120.0, 10.0, 1e-2), (200.0, 150.0, 5e-4)]:
            got = model_vector(data, vm.BinaryModelCartesian(*p))
            worst = max(worst, float(np.max(np.abs(got - disco(rec, *p)))))
        r = np.asarray(whitened_residuals(vm.BinaryModelCartesian(*TRUTH), data))
        assert np.max(np.abs(r)) < 1e-6  # whitened by sigma >= 1e-4
    record("max_abs_model_difference", worst)
    assert worst < 1e-12


@pytest.mark.validates("virgil.amigo.load_oi_data", roots=["mathematics"])
def test_one_filter_by_name(product):
    path, records = product
    data = amigo.load_oi_data(path, filter_name="F430M")
    np.testing.assert_array_equal(np.asarray(data.flatten_data()[0]), records["F430M"]["disco_coefficients"])


@pytest.mark.validates("virgil.amigo.load_oi_data", roots=["mathematics"], kind="control")
def test_stored_uv_sign_is_not_the_physical_one(product):
    """Reading the stored (u, v) as physical (no negation) mirrors the
    companion: its DISCO vector then differs from virgil's model by many
    sigma, so the documented sign flip is what the test above pins."""
    path, records = product
    data = amigo.load_oi_data(path, filter_name="F430M")
    rec = dict(records["F430M"], u=-records["F430M"]["u"], v=-records["F430M"]["v"])
    wrong = disco(rec, *TRUTH)
    got = model_vector(data, vm.BinaryModelCartesian(*TRUTH))
    assert np.max(np.abs(got - wrong) / rec["disco_sigma"]) > 10
