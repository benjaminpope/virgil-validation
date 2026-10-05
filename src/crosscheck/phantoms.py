"""Phantom sky images for the imaging benchmark, from first principles.

Nothing here imports virgil. Each family is a generic scene of the kind a
contest's *pre-submission* information described (never a published truth,
so that methods are not tuned to the answers):

- ``spotted_star``: an elliptical limb-darkened star with spots, and often a
  companion (2004: "limb darkened star with one or more spots").
- ``star_disk_planet``: an unresolved star, an inclined ring with a bright
  near-side rim, a gap, a faint outer ring and a planet (2018 readme "a young
  star's disk, with a planet"; 2024 Obj2 "a young star with a suspected
  companion").
- ``envelope``: a compact core in a clumpy extended envelope (2004: "compact
  source with extended envelope"; 2008's AGB star).
- ``spiral``: a central binary with an Archimedean dust spiral (2024 Obj1, "a
  hot star with an environment").
- ``thin_disk``: an inclined, flared disk with a dark mid-plane lane (2006).

Sizes are drawn in units of the beam, λ/B_max in mas, so that each family
stresses a coverage at its own resolution. ``sample(family, rng, beam)``
returns a parameter dict; ``render(params, npix, pixel)`` a unit-sum image
in the convention of ``sky.pixel_image`` (row 0 North, column 0 East),
supersampled to avoid aliasing; ``extent(params)`` the radius (mas) that
holds the emission.
"""

import numpy as np

FAMILIES = ("spotted_star", "star_disk_planet", "envelope", "spiral", "thin_disk")


def _logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _pa_offset(pa_deg, sep):
    pa = np.radians(pa_deg)
    return sep * np.sin(pa), sep * np.cos(pa)


def sample(family, rng, beam):
    """Random parameters (mas, degrees) for one phantom of ``family``."""
    if family == "spotted_star":
        diam = _logu(rng, 1.5, 4.0) * beam
        spots = [
            {"r": rng.uniform(0.0, 0.7), "pa": rng.uniform(0, 360), "size": rng.uniform(0.1, 0.3) * diam,
             "contrast": _logu(rng, 1.5, 4.0)}
            for _ in range(rng.integers(1, 4))
        ]
        companion = None
        if rng.random() < 0.7:
            companion = {"sep": rng.uniform(1.5, 4.0) * diam / 2 + diam / 2, "pa": rng.uniform(0, 360),
                         "ratio": _logu(rng, 0.02, 0.3), "diam": 0.1 * diam}
        return {"family": family, "diam": diam, "axis_ratio": rng.uniform(0.6, 1.0), "pa": rng.uniform(0, 180),
                "u_ld": rng.uniform(0.2, 0.8), "spots": spots, "companion": companion}
    if family == "star_disk_planet":
        r1 = _logu(rng, 0.7, 2.0) * beam
        return {"family": family, "star_frac": rng.uniform(0.4, 0.8), "star_diam": 0.2 * beam,
                "r_inner": r1, "r_outer": rng.uniform(2.5, 5.0) * r1, "inc": rng.uniform(20, 70),
                "pa": rng.uniform(0, 180), "rim": rng.uniform(0.0, 0.8), "outer_frac": rng.uniform(0.05, 0.3),
                "planet": {"sep": rng.uniform(1.0, 4.0) * beam, "pa": rng.uniform(0, 360),
                           "frac": rng.uniform(0.01, 0.05)}}
    if family == "envelope":
        radius = _logu(rng, 2.0, 6.0) * beam
        blobs = []
        for _ in range(rng.integers(3, 7)):
            r, pa = radius * np.sqrt(rng.random()), rng.uniform(0, 360)
            blobs.append({"dx": _pa_offset(pa, r)[0], "dy": _pa_offset(pa, r)[1],
                          "fwhm": _logu(rng, 0.5, 3.0) * beam, "weight": rng.uniform(0.2, 1.0)})
        return {"family": family, "core_frac": rng.uniform(0.2, 0.6), "core_fwhm": 0.3 * beam,
                "radius": radius, "halo_frac": rng.uniform(0.1, 0.4), "blobs": blobs}
    if family == "spiral":
        return {"family": family, "first_turn": _logu(rng, 1.5, 4.0) * beam, "turns": rng.uniform(1.5, 3.0),
                "width": rng.uniform(0.3, 0.6) * beam, "decay": rng.uniform(0.5, 1.5),
                "rotation": rng.uniform(0, 360), "sense": int(rng.choice([-1, 1])),
                "spiral_frac": rng.uniform(0.3, 0.7), "binary_sep": 0.3 * beam, "binary_pa": rng.uniform(0, 360)}
    if family == "thin_disk":
        return {"family": family, "scale": _logu(rng, 1.0, 4.0) * beam, "inc": rng.uniform(60, 85),
                "pa": rng.uniform(0, 180), "lane": rng.uniform(0.3, 0.9), "flare": rng.uniform(0.1, 0.3),
                "star_frac": rng.uniform(0.0, 0.3)}
    raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")


def extent(p):
    """Radius (mas) containing the phantom's emission, plus a margin."""
    f = p["family"]
    if f == "spotted_star":
        r = p["diam"] / 2
        if p["companion"]:
            r = max(r, p["companion"]["sep"] + p["companion"]["diam"])
        return 1.2 * r
    if f == "star_disk_planet":
        return 1.2 * max(p["r_outer"] * 1.25, p["planet"]["sep"])
    if f == "envelope":
        return 1.2 * (p["radius"] + 2 * max(b["fwhm"] for b in p["blobs"]))
    if f == "spiral":
        return 1.2 * (p["first_turn"] * p["turns"] + 2 * p["width"])
    return 1.2 * 4 * p["scale"]


def _gauss(x, y, dx, dy, fwhm):
    s = fwhm / 2.3548200450309493
    return np.exp(-0.5 * ((x - dx) ** 2 + (y - dy) ** 2) / s**2)


def _disk(x, y, dx, dy, diam):
    return (((x - dx) ** 2 + (y - dy) ** 2) <= (diam / 2) ** 2).astype(float)


def _frame(x, y, pa_deg):
    """Coordinates along and across an axis at position angle pa (E of N)."""
    pa = np.radians(pa_deg)
    return x * np.sin(pa) + y * np.cos(pa), x * np.cos(pa) - y * np.sin(pa)


def _brightness(p, x, y):
    f = p["family"]
    if f == "spotted_star":
        along, across = _frame(x, y, p["pa"])
        a, b = p["diam"] / 2, p["axis_ratio"] * p["diam"] / 2
        r2 = (along / a) ** 2 + (across / b) ** 2
        inside = r2 < 1
        mu = np.sqrt(np.clip(1 - r2, 0, None))
        star = np.where(inside, 1 - p["u_ld"] * (1 - mu), 0.0)
        for s in p["spots"]:
            sx, sy = _pa_offset(s["pa"], s["r"] * a)
            star = star * np.where(inside, 1 + (s["contrast"] - 1) * _gauss(x, y, sx, sy, s["size"]), 1.0)
        star = star / star.sum()
        if p["companion"]:
            c = p["companion"]
            cx, cy = _pa_offset(c["pa"], c["sep"])
            comp = _disk(x, y, cx, cy, max(c["diam"], 1e-9))
            if comp.sum() == 0:  # smaller than a subpixel: the nearest one
                comp = np.zeros_like(x)
                comp[np.unravel_index(np.argmin((x - cx) ** 2 + (y - cy) ** 2), x.shape)] = 1
            star = star + c["ratio"] * comp / comp.sum()
        return star
    if f == "star_disk_planet":
        along, across = _frame(x, y, p["pa"])
        cosi = np.cos(np.radians(p["inc"]))
        r = np.hypot(along, across / cosi)  # deprojected radius
        az = np.arctan2(across / cosi, along)
        inner = np.exp(-0.5 * ((r - p["r_inner"]) / (0.15 * p["r_inner"])) ** 2)
        inner = inner * (1 + p["rim"] * np.sin(az))  # near side brighter
        outer = np.exp(-0.5 * ((r - p["r_outer"]) / (0.2 * p["r_outer"])) ** 2)
        star = _disk(x, y, 0, 0, max(p["star_diam"], 1e-9))
        if star.sum() == 0:
            star = _gauss(x, y, 0, 0, p["star_diam"])
        px, py = _pa_offset(p["planet"]["pa"], p["planet"]["sep"])
        planet = _gauss(x, y, px, py, 0.1 * p["r_inner"])
        disk_frac = 1 - p["star_frac"] - p["planet"]["frac"]
        out = p["star_frac"] * star / star.sum() + p["planet"]["frac"] * planet / planet.sum()
        out = out + disk_frac * ((1 - p["outer_frac"]) * inner / inner.sum() + p["outer_frac"] * outer / outer.sum())
        return out
    if f == "envelope":
        core = _gauss(x, y, 0, 0, p["core_fwhm"])
        halo = _gauss(x, y, 0, 0, 2 * p["radius"])
        clumps = sum(b["weight"] * _gauss(x, y, b["dx"], b["dy"], b["fwhm"]) for b in p["blobs"])
        rest = 1 - p["core_frac"] - p["halo_frac"]
        return p["core_frac"] * core / core.sum() + p["halo_frac"] * halo / halo.sum() + rest * clumps / clumps.sum()
    if f == "spiral":
        theta = np.linspace(0, 2 * np.pi * p["turns"], 400)
        r = p["first_turn"] * theta / (2 * np.pi)
        ang = p["sense"] * theta + np.radians(p["rotation"])
        w = np.exp(-theta / (2 * np.pi * p["decay"]))
        arm = np.zeros_like(x)
        for ri, ai, wi in zip(r, ang, w):
            arm += wi * _gauss(x, y, ri * np.sin(ai), ri * np.cos(ai), p["width"])
        bx, by = _pa_offset(p["binary_pa"], p["binary_sep"] / 2)
        stars = _gauss(x, y, bx, by, 0.05 * p["first_turn"]) + 0.5 * _gauss(x, y, -bx, -by, 0.05 * p["first_turn"])
        return (1 - p["spiral_frac"]) * stars / stars.sum() + p["spiral_frac"] * arm / arm.sum()
    # thin_disk
    along, across = _frame(x, y, p["pa"])
    cosi = np.cos(np.radians(p["inc"]))
    r = np.hypot(along, across / max(cosi, 0.05))
    height = p["flare"] * r + 0.05 * p["scale"]
    disk = np.exp(-r / p["scale"]) * (1 - p["lane"] * np.exp(-0.5 * (across / (0.5 * height + 1e-9)) ** 2))
    disk = disk / disk.sum()
    if p["star_frac"] > 0:
        star = _gauss(x, y, 0, 0, 0.1 * p["scale"])
        return (1 - p["star_frac"]) * disk + p["star_frac"] * star / star.sum()
    return disk


def render(p, npix, pixel, supersample=3):
    """The phantom on an ``npix`` grid of ``pixel`` mas (row 0 North, column 0
    East), averaged over ``supersample``² sub-pixels, unit sum."""
    n = npix * supersample
    c = (n - 1) / 2.0
    sub = pixel / supersample
    east = (c - np.arange(n)) * sub
    north = (c - np.arange(n)) * sub
    x, y = np.meshgrid(east, north)
    fine = np.clip(_brightness(p, x, y), 0.0, None)
    img = fine.reshape(npix, supersample, npix, supersample).sum(axis=(1, 3))
    return img / img.sum()
