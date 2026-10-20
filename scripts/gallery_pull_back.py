"""Turn the OzSTAR `gallery_orbits` run folders into gallery summaries and rebuild the orbit gallery data.

    python scripts/gallery_pull_back.py RESULTS_DIR [--summaries DIR] [--no-render]

RESULTS_DIR is anywhere above the per-system folders of ``scripts/gallery_orbits.py`` (each holds ``gallery.json`` and
``run/summary.json``, the OrbitPipeline run folder). For every system the newest folder wins. Writes
``trust/campaigns/orbits/trackb-<slug>.json`` in the shape ``orbit_gallery`` reads (the MAP orbit of the winning alias
band, the other bands as modes, the per-night positions, the flagged nights), plus a ``blind`` block: the band
ranking, the pipeline's checks and ``table`` rows for the tile. Then ``orbit_gallery.render`` rebuilds the page data.

The published orbit enters only here, after the fit: the table compares the winning band's orbit with the paper's own
measured positions on the sky (never raw omega/Omega), and says which band holds the published period. A system with
no run, or whose run failed, gets no summary: its tile stays "awaiting" and ``--report`` lists why.

Needs no JAX.
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import orbit_gallery as G  # noqa: E402

TRACKB = REPO / "oidb" / "trackb"
EXTRA = REPO / "oidb" / "gallery_extra"  # records of targets whose gallery tile exists already (Gl 229 B)
REFERENCES = REPO / "oidb" / "references"
N_MODES = 6  # bands kept as alternative modes on the tile
ALIAS_NATS = 10.0  # a runner-up closer than this leaves the cycle count ambiguous (the pipeline's alias_nats)
SHORT_ARC_NOTE = (
    "Short-arc seeding (virgil#425) was not in this run: with few nights far apart the cycle count may be "
    "narrower than the band ranking here shows."
)

# Findings from reading a finished run by hand, shown on that system's tile (the run itself is not changed).
NOTES = {
    "hd_9312": (
        "Diagnosis (2026-10-20, from the run's own reports): the period window (3.6-2864 d) contains the published 36.5 d. "
        "The per-night position search agrees with the paper's positions on 4 of 10 nights (MJD 59191, 59498, 59819, 59907). "
        "On 5 further nights (59523, 59538, 59817, 59877, 59904) the dRA is right but the dDec is off by 3-15 mas (a north-south "
        "sidelobe; fitted separations 4.3-13.4 mas against the orbit's maximum of about 5.5 mas), and no night flag fired on them. "
        "Raw chi2/N is 22-104 on every night, so the likelihood was tempered by 1/854 and the orbit-versus-single-star evidence "
        "is only 2.4 nats. The long-period winner is therefore driven by wrong night positions, not by a missed window."
    ),
}


def run_folders(results):
    """{slug: folder} of the newest ``gallery.json`` + ``run/summary.json`` per system under ``results``."""
    found = {}
    for g in pathlib.Path(results).rglob("gallery.json"):
        s = g.parent / "run" / "summary.json"
        if s.exists() and (g.parent.name not in found or s.stat().st_mtime > (found[g.parent.name] / "run" / "summary.json").stat().st_mtime):
            found[g.parent.name] = g.parent  # newest by summary.json mtime, not by path order
    return found


def _elements(best, t_ref):
    return dict(period_day=best["period"], t_peri_mjd=t_ref + best["dt_peri"], ecc=best["ecc"], inc_deg=best["inc"],
                omega_deg=best["omega"], Omega_deg=best["Omega"], a_mas=best["a_mas"])


def _bands(run):
    """[(log_z, band json)] of the finite bands, best first."""
    out = []
    for f in sorted((run / "stages" / "search" / "bands").glob("band_*.json")):
        b = json.loads(f.read_text())
        if b.get("best") and math.isfinite(b.get("log_z", float("nan"))):
            out.append((b["log_z"], b))
    return sorted(out, key=lambda x: -x[0])


def published_orbit_epochs(record, mjds):
    """Positions of the headline published orbit at ``mjds`` for a record with a ``collection_id`` and ``pub`` (a target
    whose paper gives an orbit, not positions): ``[(mjd, dra, ddec)]``, and the orbit's period; ``([], None)`` for any
    other record. Read after the fit, never as a start."""
    if not record.get("pub"):
        return [], None
    ref = json.loads((REFERENCES / f"{record['collection_id']}.json").read_text())
    el = G.published_orbits(record["pub"], ref)[0][1]
    dra, ddec = (np.ravel(x) for x in G._orbit_at(el, np.asarray(mjds, float)))
    return [(float(m), float(a), float(d)) for m, a, d in zip(mjds, dra, ddec)], el["period_day"]


def sky_rows(best, t_ref, record, band_rows, p_range, flipped, labels_free, chi2_n, pub_orbit=None):
    """Table rows and the verdict: the winning orbit against the paper's measured positions, and the paper's period
    against the band ranking. ``band_rows`` are all the bands with a finite evidence (not the short table of the pipeline's
    summary), best first. The reference is used here only."""
    from crosscheck import skyplane as SP

    rows, verdict = [], None
    pub = [(e["mjd"], e["dra_mas"], e["ddec_mas"]) for e in record.get("epochs", []) if e.get("mjd")]
    what = "the paper's positions"
    p_pub = (record.get("orbit") or {}).get("P_day", [None])[0]
    if pub_orbit and pub_orbit[0]:  # an orbit-only paper: its orbit's positions at our nights
        pub, p_pub = pub_orbit
        what = "the published orbit's positions at our nights"
    if pub:
        mjd, dra, ddec = (np.array(c, float) for c in zip(*pub))
        ours = SP.fit_pos(best, mjd, t_ref)
        target = np.stack([dra, ddec], -1)
        rms = {s: float(np.sqrt(np.mean(np.sum((ours - s * target) ** 2, -1)))) for s in (1, -1)}
        sign = 1 if rms[1] <= rms[-1] else -1
        mean_sep = float(np.mean(np.hypot(dra, ddec)))
        verdict = SP.verdict_from_stats(rms[sign] / mean_sep, None, sign < 0, flipped, labels_free, chi2_n)
        rows.append(dict(quantity=f"sky rms vs {what} (mas)", ours=rms[sign], err=None, pub=0.0, pub_err=None,
                         pull=None, status="REPORT",
                         note=f"{verdict}; {len(pub)} epochs; {rms[sign] / mean_sep:.1%} of the mean separation; "
                              f"{'point reflection' if sign < 0 else 'same orientation'}"))
    if p_pub:
        hit = next((r for r in band_rows if r["p_lo"] <= p_pub <= r["p_hi"]), None)
        if hit:
            where = f"band N={hit['n']} (rank {hit['rank'] + 1} of {len(band_rows)}, Δlog Z {hit['delta_log_z']:+.1f})"
        elif p_range[0] <= p_pub <= p_range[1]:
            where = "a searched band that gave no finite evidence"
        else:
            where = f"no band: it is outside the searched window {p_range[0]:.3g}-{p_range[1]:.3g} d"
        rows.append(dict(quantity="period (d)", ours=best["period"], err=None, pub=p_pub, pub_err=None, pull=None,
                         status="REPORT", note="the published period lies in " + where))
    return rows, verdict


def summary_of(folder, record):
    """The gallery summary of one run folder (see the module docstring)."""
    run = folder / "run"
    s = json.loads((run / "summary.json").read_text())
    meta = json.loads((folder / "gallery.json").read_text())
    search = json.loads((run / "stages" / "search" / "report.json").read_text())
    epochs = json.loads((run / "stages" / "epochs" / "report.json").read_text())
    t_ref = search["t_ref"]
    bands = _bands(run)
    if not bands:
        raise ValueError("no band with a finite evidence")
    top = bands[0][0]
    winner = bands[0][1]
    if int(winner["n"]) != int(s["modes"]["winner"]["n"]):  # the tile must show the pipeline's own winner
        raise ValueError(f"best band file N={winner['n']} is not the pipeline's winner N={s['modes']['winner']['n']}")
    raw_chi2 = s["chi2"]["best_orbit_reduced"]  # raw chi2/N on the quoted errors at the best orbit, as the pipeline reports it
    table = s["modes"]["table"]
    positions, checks = [], []
    for n in epochs["nights"]:
        positions.append(dict(name=n["dataset"], mjd=n["mjd"], dra=n["dra_mas"], ddec=n["ddec_mas"], cov=None,
                              flux=n["flux"], note="; ".join(f["code"] for f in n["flags"]) or None,
                              alias=any(f["code"] in ("aliased-peaks", "low-gap-marginal") for f in n["flags"]) or None))
        if n["flags"]:
            checks.append(dict(name=n["dataset"], reasons=[f"flagged: {f['code']}: {f['reason']}" for f in n["flags"]]))
    orbit = dict(t_ref=t_ref, map=_elements(winner["best"], t_ref), loss=-top, start="blind",
                 chi2_raw=raw_chi2, epoch_checks=checks, positions=positions,
                 modes=[dict(loss=-z, elements=_elements(b["best"], t_ref)) for z, b in bands[:N_MODES]])
    band_rows = [dict(rank=i, n=b["n"], p_lo=b["p_lo"], p_hi=b["p_hi"], delta_log_z=z - top) for i, (z, b) in enumerate(bands)]
    pub_orbit = published_orbit_epochs(record, [p["mjd"] for p in positions])
    rows, verdict = sky_rows(winner["best"], t_ref, record, band_rows, search["p_range"], bool(winner.get("flipped")),
                             bool(winner.get("labels_free")), raw_chi2, pub_orbit)
    flags = []
    delta = s["modes"]["delta_log_z"]  # winner minus runner-up, so >= 0 (the band table's own column is <= 0)
    if delta is not None and abs(delta) < ALIAS_NATS:
        flags.append(dict(key="alias", text=f"Alias not resolved: the runner-up band is only {delta:.1f} nats behind "
                                            f"(N={s['modes']['runner_up']['n']} against the winner N={s['modes']['winner']['n']})."))
    if s["modes"]["no_orbit"]["verdict"] != "orbit":
        flags.append(dict(key="no-orbit", text=f"The orbit is not preferred over a single star ({s['modes']['no_orbit']['verdict']})."))
    # Solved problem: unconverged NUTS reported as a result. Every failed or warned check is listed on the tile.
    for status, word in (("fail", "failed"), ("warn", "warn")):
        names = [c["name"] for c in s["checks"] if c["status"] == status]
        if names:
            flags.append(dict(key=f"checks-{status}", text=f"Pipeline checks {word}: " + ", ".join(names) + "."))
    if s.get("status") != "complete":  # e.g. the CPU job aborted in NUTS (LLVM section-memory error) after the band search
        flags.append(dict(key="incomplete", text=f"The pipeline run did not finish (status {s.get('status')!r}): the band search is "
                                                 "reported, the posterior is not. Rerunning on a GPU node."))
    if record["slug"] in NOTES:
        flags.append(dict(key="note", text=NOTES[record["slug"]]))
    flags.append(dict(key="short-arc", text=SHORT_ARC_NOTE))
    blind = dict(
        alias_bands=table, flags=flags, table=rows, verdict=verdict, p_range=search["p_range"], n_bands=search["n_bands"],
        flagged_nights=[c["name"] for c in checks], checks=[dict(name=c["name"], status=c["status"]) for c in s["checks"]],
        no_orbit=s["modes"]["no_orbit"], primary_diam=search["primary_diam"], pipeline_status=s.get("status"),
    )
    return dict(collection_id=f"trackb-{record['slug']}", target=record["target"],
                instrument=G.INSTRUMENT.get(meta["instrument"], meta["instrument"]),
                provenance=dict(virgil_commit=meta.get("virgil_commit"), validation_commit=meta.get("validation_commit"),
                                virgil_vlti_commit=meta.get("virgil_vlti_commit"),
                                # absent for a run that predates it: the tile's legacy flag then stays
                                **({k: meta[k] for k in ("data_path",) if meta.get(k)}),
                                night_paths={k: v.get("data_path") for k, v in (meta.get("nights") or {}).items()
                                             if isinstance(v, dict)} or None),
                orbit=orbit, blind=blind)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=pathlib.Path)
    ap.add_argument("--summaries", type=pathlib.Path, default=G.SUMMARIES)
    ap.add_argument("--no-render", action="store_true", help="write the summaries, skip the page data")
    a = ap.parse_args(argv)
    folders = run_folders(a.results)
    if not folders:
        sys.exit(f"no <system>/gallery.json with run/summary.json under {a.results}")
    a.summaries.mkdir(parents=True, exist_ok=True)
    failed = {}
    records = {p.stem: p for d in (TRACKB, EXTRA) for p in d.glob("*.json")}
    for slug in sorted(records):
        if slug not in folders:
            failed[slug] = "no finished run folder"
            continue
        try:
            summary = summary_of(folders[slug], json.loads(records[slug].read_text()))
        except Exception as err:  # noqa: BLE001 a failed target is listed, not hidden
            failed[slug] = f"{type(err).__name__}: {err}"
            continue
        dest = a.summaries / f"{summary['collection_id']}.json"  # never the committed T1-T5 summary of the collection (oidb_compare.compare reads that shape)
        dest.write_text(json.dumps(G._clean(summary), indent=1) + "\n")
        print(f"wrote {dest.relative_to(REPO) if dest.is_relative_to(REPO) else dest}")
    for slug, why in failed.items():
        print(f"NOT PULLED {slug}: {why}")
    if not a.no_render:
        G.render(REPO / "docs" / "orbits" / "binaries.md", REPO / "docs" / "assets" / "orbits.json", a.summaries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
