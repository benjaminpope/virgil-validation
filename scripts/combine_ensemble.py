"""Combine the ensemble members of the contest campaign (PYRA-style).

    python scripts/combine_ensemble.py RESULTS_DIR [RESULTS_DIR ...] [--out DIR]

With several directories (e.g. a campaign and a rerun of some of its
members), a member in a later one replaces the same member in an earlier one.

For each dataset, read the members' `<label>_m<k>.npz` files, written by
`scripts/contest_images.py --member k`. Each member is a CLEAN-started GP fit
with a randomised start, rendered onto a common grid (twice the base field,
129 pixels). Then:

1. **χ² filter.** Keep the members whose χ²/N, with the quoted errors, is
   within 1.5x of the best member's, or of 1 if the best is below 1. This
   drops starts that stayed in a poor basin, as PYRA/MYTHRA and the 2012
   random-start entries did. The floor at 1 stops a member that over-fits
   (χ²/N ≪ 1, e.g. 2022 AMI at 0.02) from rejecting the ones that fit to
   the noise.
2. **Average.** Take the mean and the per-pixel standard deviation of the
   kept members' unit-sum images. A feature that is real survives the
   average; one that comes from a single start's artefact is diluted and
   shows up in σ.
3. **Star or no star.** Compare the best log evidence among star members
   with the best among no-star members: same data, same errors.

Writes `<label>_ensemble.npz` and `<label>_ensemble.png`, plus `ensemble.md`,
a table over all datasets. NumPy and matplotlib only, so it runs on a laptop.
"""

import argparse
import collections
import pathlib
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

KEEP = 1.5  # members within this factor of max(best χ²/N, 1) are kept


def load(*results):
    latest = {}  # file name -> path; later directories win
    for directory in results:
        for path in pathlib.Path(directory).glob("*_m[0-9]*.npz"):
            latest[path.name] = path
    groups = collections.defaultdict(list)
    for name, path in sorted(latest.items()):
        match = re.match(r"(.+)_m(\d+)\.npz$", name)
        if not match:
            continue
        d = np.load(path)
        if "ref_image" not in d:
            continue
        groups[match.group(1)].append({
            "member": int(match.group(2)),
            "image": np.asarray(d["ref_image"], float),
            "fov": float(d["ref_fov"]),
            "star": bool(d["star"]),
            "log_z": float(d["best_log_z"]),
            "chi2": float(d["best_chi2_red"]),
            "scale": float(d["error_scale"]),
            "flip": float(d["flip_dchi2"]),
        })
    return groups


def resample(image, fov, common_fov, npix=None):
    """``image`` (unit-sum, ``fov`` across) on an ``npix`` grid (default: its
    own size) spanning ``common_fov``, by bilinear interpolation, renormalised
    to unit sum. Members of different campaigns can differ in both field and
    pixel count (job 18073823 used 129 pixels over 2x the base field, later
    campaigns 257 over 4x); resampling makes them comparable pixel by pixel."""
    from scipy.ndimage import gaussian_filter, map_coordinates

    n = image.shape[0]
    npix = n if npix is None else int(npix)
    # Size of a new pixel in old pixels.
    ratio = (common_fov / npix) / (fov / n)
    if ratio > 1:
        # Larger new pixels: smooth over about one of them first, so that
        # point sampling cannot step over a compact feature (anti-aliasing).
        image = gaussian_filter(image, sigma=0.5 * ratio, mode="constant")
    # Pixel k of the new grid sits at (k - c_new) * common_fov / npix mas from
    # the centre, which is pixel c_old + that / (fov / n) of the old one.
    k = (n - 1) / 2 + (np.arange(npix) - (npix - 1) / 2) * ratio
    rows, cols = np.meshgrid(k, k, indexing="ij")
    out = map_coordinates(image, [rows, cols], order=1, mode="constant", cval=0.0)
    return out / out.sum()


def combine(label, members, out):
    # Members of one campaign share a grid (contest_images.reference_fov);
    # members of different campaigns are resampled onto the largest field
    # and the largest pixel count among them.
    common = max(m["fov"] for m in members)
    npix = max(m["image"].shape[0] for m in members)
    for m in members:
        if not np.isclose(m["fov"], common, rtol=1e-9) or m["image"].shape[0] != npix:
            m["image"] = resample(m["image"] / m["image"].sum(), m["fov"], common, npix)
            m["fov"] = common
    chi2 = np.array([m["chi2"] for m in members])
    finite = np.isfinite(chi2)
    best = np.nanmin(chi2[finite]) if finite.any() else np.nan
    cutoff = KEEP * max(best, 1.0)  # an over-fitting best member must not set the bar
    kept = [m for m, c in zip(members, chi2) if np.isfinite(c) and c <= cutoff]
    stack = np.array([m["image"] / m["image"].sum() for m in kept])
    mean, sigma = stack.mean(0), stack.std(0)
    star = [m["log_z"] for m in members if m["star"] and np.isfinite(m["log_z"])]
    nostar = [m["log_z"] for m in members if not m["star"] and np.isfinite(m["log_z"])]
    dlogz = (max(star) - max(nostar)) if star and nostar else np.nan
    fov = kept[0]["fov"]
    np.savez_compressed(out / f"{label}_ensemble.npz", mean=mean, sigma=sigma, fov=fov,
                        kept=np.array([m["member"] for m in kept]),
                        members=np.array([m["member"] for m in members]), chi2=chi2, star_minus_nostar_log_z=dlogz)

    n = len(members)
    fig, axes = plt.subplots(1, 2 + n, figsize=(3.2 * (2 + n), 3.4))
    extent = [fov / 2, -fov / 2, -fov / 2, fov / 2]  # East left, North up
    for ax, img, title in ((axes[0], mean, f"mean of {len(kept)}/{n}"), (axes[1], sigma, "σ")):
        ax.imshow(img, origin="upper", extent=extent, cmap="magma")
        ax.set(title=title, xlabel="ΔRA (mas)", ylabel="ΔDec (mas)")
    for ax, m in zip(axes[2:], sorted(members, key=lambda m: m["member"])):
        ax.imshow(m["image"], origin="upper", extent=extent, cmap="magma")
        mark = "" if any(k is m for k in kept) else " (dropped)"
        ax.set(title=f"m{m['member']} {'star' if m['star'] else 'no star'}\nχ²/N {m['chi2']:.3g}{mark}", xticks=[], yticks=[])
    fig.suptitle(label)
    plt.tight_layout()
    fig.savefig(out / f"{label}_ensemble.png", dpi=90)
    plt.close(fig)
    scales = [m["scale"] for m in kept]
    return (f"| {label} | {len(kept)}/{n} | {best:.3g} | {np.median([m['chi2'] for m in kept]):.3g} | "
            f"{min(scales):.2f}–{max(scales):.2f} | {dlogz:+.4g} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = pathlib.Path(args.out or args.results[-1])
    out.mkdir(parents=True, exist_ok=True)
    rows = ["| dataset | kept | best χ²/N | median kept χ²/N | error scale (kept) | ΔlogZ star − no star |",
            "|---|---|---|---|---|---|"]
    for label, members in sorted(load(*args.results).items()):
        rows.append(combine(label, members, out))
    table = "\n".join(rows) + "\n"
    (out / "ensemble.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
