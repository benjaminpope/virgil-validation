"""Published reference orbits, with omega normalised by an explicit ``omega_of`` field.

The reference files in ``oidb/references/*.json`` record what papers published. Papers disagree about whose
argument of periastron they tabulate: Xuan et al. 2024 (Gl 229) quote the primary's, Klement et al. 2025
(HR 6819) the companion's, and some papers do not say. The projected orbit depends on it: the primary's omega is
the secondary's + 180 deg, which negates every relative position (a point reflection, the label swap). Guessing
this per call site is how the same convention error was re-found several times, so every reference orbit now
carries a **required** field

``omega_of`` : ``"primary"`` or ``"secondary"``
    Whose argument of periastron the tabulated omega is.

and `secondary_omega` is the only place a tabulated omega is converted to the companion's omega that
`crosscheck.orbits` (and so `crosscheck.skyplane`) use.

Solved problems handled here:

* primary versus secondary omega: `secondary_omega` adds 180 deg for ``"primary"``; a missing or unknown
  ``omega_of`` is an error, never a default;
* the (Omega + 180, omega + 180) twin is one sky orbit and needs no conversion; it is handled by comparing in the
  sky plane (`crosscheck.skyplane`), not here;
* Track B ``system.json`` files (kept outside the repository, schema ``reference_orbit`` with P_day, T0, e,
  omega_deg, Omega_deg, i_deg, a_mas) are read by `trackb_reference`, with ``omega_of`` taken from the file's
  ``reference_orbit`` or given by the caller.

This module never imports virgil.
"""

import glob
import json
import pathlib

OMEGA_OF = ("primary", "secondary")
MJD_JD = 2400000.5  # MJD = JD - 2400000.5
_OMEGA_KEYS = ("omega_deg", "omega_primary_deg")

REFERENCE_DIR = pathlib.Path(__file__).resolve().parents[2] / "oidb" / "references"


def secondary_omega(omega, omega_of):
    """Convert a tabulated argument of periastron to the secondary's (companion's), in [0, 360).

    Solved problem, primary versus secondary omega: the primary's omega is the secondary's + 180 deg, so a primary's
    omega read as the secondary's gives the point reflection of the orbit. The caller must say which it is.

    Parameters
    ----------
    omega : float
        The tabulated omega, degrees.
    omega_of : {"primary", "secondary"}
        Whose omega it is.

    Returns
    -------
    float
        The secondary's omega, degrees, in [0, 360).

    Raises
    ------
    ValueError
        If ``omega_of`` is not ``"primary"`` or ``"secondary"`` (including None: there is no default).
    """
    if omega_of not in OMEGA_OF:
        raise ValueError(f"omega_of must be one of {OMEGA_OF}, got {omega_of!r}")
    return float((omega + (180.0 if omega_of == "primary" else 0.0)) % 360.0)


def reference_files(directory=None):
    """Paths of the reference JSON files, sorted.

    Parameters
    ----------
    directory : path-like, optional
        Folder to read (default ``oidb/references`` of this repository).

    Returns
    -------
    list of pathlib.Path
    """
    return [pathlib.Path(p) for p in sorted(glob.glob(str(pathlib.Path(directory or REFERENCE_DIR) / "*.json")))]


def load_references(directory=None):
    """Load every reference file.

    Parameters
    ----------
    directory : path-like, optional
        Folder to read (default ``oidb/references``).

    Returns
    -------
    dict
        ``{file stem: parsed JSON}``.
    """
    return {p.stem: json.loads(p.read_text()) for p in reference_files(directory)}


def orbit_blocks(doc, path=""):
    """Every reference orbit in a reference document, wherever it is nested.

    A reference orbit is a dict carrying ``omega_deg`` or ``omega_primary_deg``, or a direct child of an
    ``all_solutions`` dict (a table of published solutions whose rows may be lists, which cannot carry a field).

    Parameters
    ----------
    doc : dict or list
        A parsed reference JSON (or any nested part of one).
    path : str
        Path prefix of ``doc``, for the returned labels.

    Returns
    -------
    list of (str, dict)
        ``(path, block)`` pairs, in document order.
    """
    found = []
    if isinstance(doc, dict):
        if any(k in doc for k in _OMEGA_KEYS) or _is_solution_child(path):
            found.append((path or "/", doc))
        for k, v in doc.items():
            if isinstance(v, dict):
                found += orbit_blocks(v, f"{path}/{k}")
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            found += orbit_blocks(v, f"{path}[{i}]")
    return found


def _is_solution_child(path):
    """True if ``path`` names a direct child of an ``all_solutions`` dict."""
    parts = path.split("/")
    return len(parts) >= 2 and parts[-2] == "all_solutions"


def missing_omega_of(doc):
    """Problems with the ``omega_of`` fields of one reference document.

    Parameters
    ----------
    doc : dict
        A parsed reference JSON.

    Returns
    -------
    list of str
        One message per orbit block whose ``omega_of`` is missing, not ``"primary"``/``"secondary"``, or
        contradicts an ``omega_primary_deg`` key. Empty when the document is valid.
    """
    problems = []
    for path, block in orbit_blocks(doc):
        value = block.get("omega_of")
        if value not in OMEGA_OF:
            problems.append(f"{path}: omega_of is {value!r}, must be one of {OMEGA_OF}")
        elif "omega_primary_deg" in block and value != "primary":
            problems.append(f"{path}: has omega_primary_deg but omega_of is {value!r}")
    return problems


def trackb_reference(system, omega_of=None):
    """The reference orbit of a Track B ``system.json``, in `crosscheck.orbits` convention.

    Parameters
    ----------
    system : dict
        Parsed ``system.json`` with ``reference_orbit`` (P_day, T0, e, omega_deg, Omega_deg, i_deg, a_mas, each
        ``[value, ...]``; T0 carries its time system as a third element) and ``reference`` (source).
    omega_of : {"primary", "secondary"}, optional
        Whose omega the table gives. Defaults to ``system["reference_orbit"]["omega_of"]``; one of the two is
        required.

    Returns
    -------
    dict or None
        P, t_peri (MJD), ecc, inc, omega (the secondary's), Omega, a_mas and source; None if the system has no
        complete reference orbit.

    Raises
    ------
    ValueError
        If no valid ``omega_of`` is given or found.
    """
    ro = system.get("reference_orbit") or {}
    need = ("P_day", "T0", "e", "omega_deg", "Omega_deg", "i_deg", "a_mas")
    if not all(k in ro for k in need):
        return None
    omega_of = omega_of if omega_of is not None else ro.get("omega_of")
    t0 = ro["T0"]
    unit = str(t0[2]) if len(t0) > 2 else ""
    if "BJD-2400000" in unit:  # MJD = BJD - 2400000.5, so BJD-2400000 less 0.5 d
        t_peri = t0[0] - 0.5
    else:
        jd = t0[0] + 2445000.0 if "HJD-2445000" in unit else t0[0]  # AL Dor: Tp is HJD-2445000
        t_peri = jd - MJD_JD
    ref = system.get("reference", {})
    return dict(
        P=ro["P_day"][0],
        t_peri=t_peri,
        ecc=ro["e"][0],
        inc=ro["i_deg"][0],
        omega=secondary_omega(ro["omega_deg"][0], omega_of),
        Omega=ro["Omega_deg"][0],
        a_mas=ro["a_mas"][0],
        source=ref.get("positions_source", "")[:60] or ref.get("bibcode", ""),
    )
