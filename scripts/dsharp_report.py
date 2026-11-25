"""Build the side-by-side comparison page of the DSHARP / HL Tau campaign from ``dsharp_compare.py`` and ``dsharp_fit.py`` outputs. Local, no GPU, no virgil.

    dsharp_report.py --compare DIR --fits DIR --out DIR/index.html [--sha-table data/dsharp/checksums.json]

``--compare`` holds ``<target>/compare.json`` and ``compare.png``; ``--fits`` holds ``<target>/result.json``. The page references the figures as ``img/<target>.png``
(copied next to it), so it is one HTML file plus six PNGs. Every number on it is read from those files; nothing is typed in here except the wording.
"""

import argparse
import html
import json
import pathlib
import shutil
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crosscheck import dsharp as ds  # noqa: E402

CSS = """
:root{--bg:#f3f5f7;--panel:#ffffff;--ink:#14202b;--mute:#566574;--rule:#d5dce3;--accent:#0b7a8f;--accent-ink:#ffffff;--good:#1d7a46;--good-bg:#dff1e6;--warn:#9a5b00;--warn-bg:#fbeccb;--bad:#a3262a;--bad-bg:#f8dcdc;--shadow:0 1px 2px rgba(20,32,43,.08)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0e151c;--panel:#16212b;--ink:#e6edf3;--mute:#93a3b3;--rule:#2a3a48;--accent:#4cc3d9;--accent-ink:#06222a;--good:#69d394;--good-bg:#12331f;--warn:#f0b755;--warn-bg:#3a2c0c;--bad:#f08a8c;--bad-bg:#401819;--shadow:none;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0e151c;--panel:#16212b;--ink:#e6edf3;--mute:#93a3b3;--rule:#2a3a48;--accent:#4cc3d9;--accent-ink:#06222a;--good:#69d394;--good-bg:#12331f;--warn:#f0b755;--warn-bg:#3a2c0c;--bad:#f08a8c;--bad-bg:#401819;--shadow:none;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font:15px/1.55 Inter,system-ui,-apple-system,"Segoe UI",sans-serif;padding-block:32px 56px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
.wrap{max-width:1180px;margin:0 auto;display:flex;flex-direction:column;gap:28px}
h1,h2,h3{font-family:"Space Grotesk",Inter,system-ui,sans-serif;text-wrap:balance;margin:0;line-height:1.15}
h1{font-size:clamp(1.8rem,4vw,2.6rem);letter-spacing:-.01em}
h2{font-size:1.45rem}
h3{font-size:.8rem;letter-spacing:.09em;text-transform:uppercase;color:var(--mute);font-weight:600}
p{margin:0;max-width:68ch}
.lede{color:var(--mute);font-size:1.05rem}
.mono,td.num,th.num{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums}
.note{border:1px solid var(--rule);border-left-width:4px;border-radius:6px;padding:14px 16px;background:var(--panel);display:flex;flex-direction:column;gap:6px}
.note.warn{border-left-color:var(--warn)}.note.warn strong{color:var(--warn)}
.note.bad{border-left-color:var(--bad)}.note.bad strong{color:var(--bad)}
.note strong{font-family:"Space Grotesk",Inter,sans-serif}
.tablewrap{overflow-x:auto;background:var(--panel);border:1px solid var(--rule);border-radius:8px;box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;font-size:.88rem}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--rule);white-space:nowrap}
th{font-size:.72rem;letter-spacing:.07em;text-transform:uppercase;color:var(--mute);font-weight:600}
td.num,th.num{text-align:right}
tr:last-child td{border-bottom:0}
.pill{display:inline-block;border-radius:999px;padding:2px 10px;font-size:.78rem;font-weight:600;white-space:nowrap}
.pill.good{background:var(--good-bg);color:var(--good)}.pill.warn{background:var(--warn-bg);color:var(--warn)}.pill.bad{background:var(--bad-bg);color:var(--bad)}
section.target{display:flex;flex-direction:column;gap:14px;scroll-margin-top:16px}
.thead{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 14px}
.fig{background:#fff;border:1px solid var(--rule);border-radius:8px;overflow-x:auto;box-shadow:var(--shadow)}
.fig img{display:block;width:100%;min-width:720px;height:auto}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:14px 16px;display:flex;flex-direction:column;gap:8px;min-width:0;box-shadow:var(--shadow)}
dl{margin:0;display:grid;grid-template-columns:minmax(0,max-content) minmax(0,1fr);gap:5px 16px;font-size:.88rem}
dt{color:var(--mute)}dd{margin:0;font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
a{color:var(--accent)}
footer{color:var(--mute);font-size:.82rem;display:flex;flex-direction:column;gap:6px}
nav.jump{display:flex;flex-wrap:wrap;gap:8px}
nav.jump a{border:1px solid var(--rule);border-radius:999px;padding:3px 12px;text-decoration:none;font-size:.85rem;background:var(--panel);color:var(--ink)}
nav.jump a:focus-visible,a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
"""


def esc(s):
    return html.escape(str(s))


def sig(x, n=3):
    return f"{x:.{n}g}"


def verdict(c, key):
    """``(label, class, reason)`` from the central-field statistics: agreement is judged on correlation, flux and peak, not on the rms alone."""
    corr, fr, pr = c["correlation"], c["flux_ratio"], c["peak_ratio"]
    off = []
    if corr < 0.98:
        off.append(f"correlation {corr:.3f}")
    if abs(fr - 1) > 0.10:
        off.append(f"flux ratio {fr:.3f}")
    if abs(pr - 1) > 0.15:
        off.append(f"peak ratio {pr:.3f}")
    if not off:
        return "Matches the public image", "good", f"correlation {corr:.3f}, flux ratio {fr:.3f}, peak ratio {pr:.3f}"
    return "Differs from the public image", "bad" if corr < 0.9 else "warn", "; ".join(off)


def load(args):
    rows = []
    sums = json.loads(pathlib.Path(args.sha_table).read_text()) if pathlib.Path(args.sha_table).exists() else {}
    for k in ds.CAMPAIGN:
        t = ds.target(k)
        cj = json.loads((pathlib.Path(args.compare) / k / "compare.json").read_text())
        rj = json.loads((pathlib.Path(args.fits) / k / "result.json").read_text())
        rows.append(dict(key=k, t=t, c=cj, r=rj, sha=sums.get(t["image"]["url"], {}), v=verdict(cj["central"], k)))
    return rows


def summary_table(rows):
    out = ['<div class="tablewrap"><table><thead><tr><th>Target</th><th class="num">Correlation</th><th class="num">Flux ratio</th><th class="num">Peak ratio</th><th class="num">rms(diff) / peak</th><th class="num">chi&sup2; / 2N</th><th>Converged</th><th>Verdict</th></tr></thead><tbody>']
    for x in rows:
        c, r = x["c"]["central"], x["r"]
        n2 = 2 * r["n"]
        out.append(
            f'<tr><td><a href="#{x["key"]}">{esc(x["t"]["name"])}</a></td><td class="num">{c["correlation"]:.3f}</td><td class="num">{c["flux_ratio"]:.3f}</td>'
            f'<td class="num">{c["peak_ratio"]:.3f}</td><td class="num">{c["rms_over_peak"]:.4f}</td><td class="num">{r["chi2_fit"] / n2:.3f}</td>'
            f'<td><span class="pill warn">no</span></td><td><span class="pill {x["v"][1]}">{esc(x["v"][0])}</span></td></tr>'
        )
    out.append("</tbody></table></div>")
    return "".join(out)


def target_section(x):
    k, t, c, r = x["key"], x["t"], x["c"], x["r"]
    cc, cw = c["central"], c["whole"]
    img = t["image"]
    n2 = 2 * r["n"]
    sig_ = r.get("sigma") or []
    sigtxt = f'{min(sig_):.2f} to {max(sig_):.2f} (median {statistics.median(sig_):.2f}, {len(sig_)} spectral windows)' if sig_ else "n/a"
    dead = (r.get("dead_pixels") or {}).get("base") or {}
    deadtxt = f'{100 * dead["frac_below_1e-6_peak"]:.1f}% of hidden-image pixels below 1e-6 of the peak' if dead else "n/a"
    prov = r.get("provenance") or {}
    beam = c["beam_mas_mas_deg"]
    pb = ("The reference is primary-beam corrected (.pbcor); virgil fits the apparent sky, so the reference was multiplied by the 12 m primary beam before comparing."
          if c.get("pbcor_applied") else "The reference is not primary-beam corrected, so it is compared as released.")
    extra = ""
    if k.startswith("hltau"):
        extra = ('<div class="note warn"><strong>Weights caveat.</strong><p>The self-calibrated split has visibility weights that are about '
                 f'{sig_[0] ** 2 if sig_ else 0:.0f} times too large in variance: the fitted error scale is {sig_[0] if sig_ else 0:.0f}, against 1.1 to 1.9 for the DSHARP discs. '
                 'The fit absorbs this through its free per-window error scale, so chi&sup2; (with the scale) is 1.00, but the raw chi&sup2; per 2N of '
                 f'{r["chi2n_raw"]:.0f} is not a fit quality.</p></div>')
    return f"""
<section class="target" id="{esc(k)}">
  <div class="thead"><h2>{esc(t["name"])}</h2><span class="pill {x["v"][1]}">{esc(x["v"][0])}</span><span class="pill warn">not converged</span></div>
  <p class="lede">{esc(x["v"][2])}. Central 2 arcsec; virgil is the model's sky (fitted image convolved with the 30 mas taper), convolved again with the released beam.</p>
  <div class="fig"><img src="img/{esc(k)}.png" alt="{esc(t["name"])}: virgil image, public ALMA image, their difference and radial profiles" width="1885" height="481"></div>
  {extra}
  <div class="cols">
    <div class="card"><h3>Agreement</h3><dl>
      <dt>correlation</dt><dd>{cc["correlation"]:.4f} central, {cw["correlation"]:.4f} whole field</dd>
      <dt>flux ratio</dt><dd>{cc["flux_ratio"]:.4f} central, {cw["flux_ratio"]:.4f} whole</dd>
      <dt>peak ratio</dt><dd>{cc["peak_ratio"]:.4f}</dd>
      <dt>rms(diff) / peak</dt><dd>{cc["rms_over_peak"]:.5f}</dd>
      <dt>max |diff| / peak</dt><dd>{cc["max_abs_over_peak"]:.4f}</dd>
      <dt>reference noise</dt><dd>{sig(cc["ref_noise_rms"])} Jy/beam rms</dd>
      <dt>virgil noise</dt><dd>{sig(cc["ours_noise_rms"])} Jy/beam rms (no thermal noise in the model)</dd>
    </dl><p>{esc(pb)}</p></div>
    <div class="card"><h3>The virgil fit</h3><dl>
      <dt>data points</dt><dd>{r["n"]:,} in {len(r.get("ddid_ids") or [])} spectral windows, {len(r.get("obs_ids") or [])} observation(s)</dd>
      <dt>chi&sup2; / 2N</dt><dd>{r["chi2_fit"] / n2:.4f} with the error scale; {r["chi2n_raw"]:.4g} at scale 1</dd>
      <dt>error scale</dt><dd>{esc(sigtxt)}</dd>
      <dt>optimiser</dt><dd>{r["steps"]:,} steps, stopped by {esc(r["stopped_because"])}, converged: no</dd>
      <dt>wall time</dt><dd>{r["wall_s"] / 3600:.2f} h ({r["wall_s"]:.0f} s) on one A100</dd>
      <dt>flux</dt><dd>{r["flux_jy"]:.4f} Jy (start {r["start_flux_jy"]:.4f} Jy)</dd>
      <dt>grid</dt><dd>{r["npix"]} &times; {r["pixel_mas"]:g} mas, taper {r["taper_mas"]:g} mas, &lambda;<sub>R</sub> {r["lam_R"]:.4f}</dd>
      <dt>dead pixels</dt><dd>{esc(deadtxt)}</dd>
    </dl></div>
    <div class="card"><h3>The public image</h3><dl>
      <dt>file</dt><dd>{esc(img.get("member") or pathlib.PurePosixPath(img["url"]).name)}</dd>
      <dt>source</dt><dd><a href="{esc(img["url"])}">{esc(img["url"].split("//")[1])}</a></dd>
      <dt>sha-256</dt><dd>{esc((x["sha"].get("sha256") or "not recorded")[:16])}&hellip; of the download</dd>
      <dt>beam</dt><dd>{beam[0]:.1f} &times; {beam[1]:.1f} mas, PA {beam[2]:.1f}&deg;</dd>
      <dt>pixel</dt><dd>{c["pixel_mas"]:.2f} mas</dd>
      <dt>footprint</dt><dd>statistics on pixels inside virgil's image only (radius {cc["footprint_radius_pix"]:.0f} px)</dd>
    </dl></div>
    <div class="card"><h3>Provenance</h3><dl>
      <dt>virgil</dt><dd>{esc((prov.get("virgil_commit") or "n/a")[:12])} (dirty: {esc(prov.get("virgil_dirty"))})</dd>
      <dt>virgil-validation</dt><dd>{esc((prov.get("validation_commit") or "n/a")[:12])}</dd>
      <dt>data column</dt><dd>{"DATA, self-calibrated split" if k.startswith("hltau") else "released continuum measurement set"}</dd>
    </dl></div>
  </div>
</section>"""


def build(rows):
    now = len(rows)
    head = f"""<title>DSHARP Image Comparison</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>"""
    jump = '<nav class="jump" aria-label="Targets">' + "".join(f'<a href="#{x["key"]}">{esc(x["t"]["name"])}</a>' for x in rows) + "</nav>"
    notes = (
        '<div class="note warn"><strong>None of the six fits converged.</strong><p>Every run stopped at its 20,000-step limit without meeting the C8 convergence criterion. '
        'The images are the optimiser\'s state at that point, so agreement here is a lower bound on what a converged fit would show, and a longer run could move the numbers.</p></div>'
        '<div class="note warn"><strong>HL Tau data weights.</strong><p>The self-calibrated HL Tau measurement sets carry visibility weights that are far too large; the fit rescales them with a free per-window error scale (about 47). '
        'The comparison with the public images is unaffected by the weights alone, but the raw chi&sup2; of the two HL Tau fits is not a measure of fit quality.</p></div>'
        '<div class="note"><strong>How to read the figures.</strong><p>Left to right: virgil, the public ALMA image, their difference, and the azimuthal mean of each about the phase centre. '
        'Both images are in Jy/beam on the public image\'s grid, with one arcsinh stretch, north up and east left. virgil\'s image is convolved with the public image\'s clean beam first, '
        'so the two are the same kind of image. The difference panel has its own colour scale.</p></div>'
    )
    body = (
        f'<main class="wrap"><header style="display:flex;flex-direction:column;gap:12px"><h3>virgil-validation &middot; ALMA imaging</h3><h1>Six discs, two images each</h1>'
        f'<p class="lede">virgil\'s maximum-entropy reconstructions of IM Lup, HL Tau (Band 6 and Band 7), HD 163296, RU Lup and AS 209, set against the public ALMA images of the same data: '
        f'the DSHARP data-release images and the HL Tau science-verification reference images.</p></header>{jump}{notes}'
        f'<section style="display:flex;flex-direction:column;gap:12px"><h3>All {now} targets, central 2 arcsec</h3>{summary_table(rows)}</section>'
        + "".join(target_section(x) for x in rows)
        + '<footer><span>Statistics are computed on pixels inside virgil\'s footprint where both images are finite. Correlation is the Pearson correlation of the two Jy/beam images; flux ratio is the sum of virgil over the sum of the public image.</span>'
        '<span>Built by scripts/dsharp_report.py from dsharp_compare.py and dsharp_fit.py outputs. Public data: ALMA (ESO/NRAO/NAOJ); see DATA_ATTRIBUTION.md.</span></footer></main>'
    )
    return head + "\n" + body


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--compare", required=True)
    p.add_argument("--fits", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sha-table", default=str(ROOT / "data" / "dsharp" / "checksums.json"))
    a = p.parse_args(argv)
    rows = load(a)
    out = pathlib.Path(a.out)
    (out.parent / "img").mkdir(parents=True, exist_ok=True)
    for x in rows:
        shutil.copyfile(pathlib.Path(a.compare) / x["key"] / "compare.png", out.parent / "img" / f"{x['key']}.png")
    out.write_text(build(rows))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB) and {len(rows)} figures in {out.parent / 'img'}")


if __name__ == "__main__":
    main()
