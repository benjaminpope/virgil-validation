"""MPoL's IM Lup procedure emulated in vv's own optax loop (Track B of ``design/alma_imlup_p3.md``: ``--workflow mpol-replica``), not in virgil core.

What MPoL's ``sgd.py`` does (mpol 0.3.1, the p3a run: ``--lr 1e-2 --FWHM 0.03 --lam-ent 1e-9 --epochs 10``, two chained stages, ``train.sbatch``):

- pixels ``I = softplus(base_cube)`` (``BaseCube``), the base cube starting at -3 (1.2837 Jy over the grid);
- the 30 mas Gaussian ``GaussConvFourier`` is a layer of the forward model, and the entropy ``losses.entropy`` (prior_intensity 1e-4, tot_flux 0.253) is of the blurred ``icube``;
- ``log10`` amplitude and sigma factors and the offsets in arcsec are the other parameters (zeros at the start), all under ``torch.optim.AdamW(lr=1e-2)`` whose weight decay is
  torch's default 1e-2 (decoupled, on every parameter, so the factors are pulled to 1 and the offsets to 0);
- the loss of a batch is ``-ln L_b / (2 n_b) + lambda_M S_M`` (``neg_log_likelihood_avg``), batches of 40,000 points, 10 epochs per stage, about 1,061 batches per epoch on all the data.

Here the same loss is taken on all the data at once (full batch, virgil's Toeplitz chi^2, the blur through virgil's ``blur_fwhm_mas``), for the same number of steps, with optax's AdamW at weight
decay 1e-2 on the same parametrisation; the parameters and the optimiser state are float32. The minibatch noise is the one difference by design. Ablations: no weight decay, the softmax
parametrisation of virgil in place of softplus, another step count.
"""

import math
import time

import numpy as np

from .alma import MPOL, ARCSEC, SB_OBS, dots_report, image_of, tapered_image

MPOL_RUN = dict(lr=1e-2, weight_decay=1e-2, batch_size=40000, epochs=10, stages=("all", "all"), base_start=-3.0, lam_m=1e-9)  # p3a's flags (train.sbatch); both of its stages ran on all the data
MPOL_STEPS = dict(sb=10 * 273, all=10 * 1061)  # IM Lup: epochs x ceil(N / batch_size), N = 10,883,819 (obs 0-4) and 42,421,131 (p3a summary.md)
REPLICA_PASS = dict(correlation=0.99, n_eff_ratio=0.78, n_eff_ratio_tol=0.20, peak_base=10.0, peak_factor=2.0, flux=0.294, flux_tol=0.05)  # Ben's pass, 2026-10-28


def steps_for(n_points, epochs=MPOL_RUN["epochs"], batch=MPOL_RUN["batch_size"]):
    """MPoL's steps for ``n_points`` per stage: ``epochs * ceil(n / batch)``."""
    return epochs * math.ceil(n_points / batch)


def mpol_entropy(cube, prior_intensity=MPOL["prior"], tot_flux=MPOL["f_tot"], floor=1e-30):
    """MPoL's ``losses.entropy``: ``(1 / tot_flux) sum I ln(I / p)`` of a cube of intensities (Jy/arcsec^2), in jax. Values are floored at ``floor`` (the blur's rounding noise can go below 0)."""
    import jax.numpy as jnp

    i = jnp.maximum(cube, floor)
    return jnp.sum(i * jnp.log(i / prior_intensity)) / tot_flux


def gauss_blur(image, fwhm_mas, pixel_mas):
    """``image`` convolved with a circular Gaussian of FWHM ``fwhm_mas`` through the FFT grid (the Fourier multiplier of MPoL's ``GaussConvFourier``), in jax; ``fwhm_mas`` 0 or None: unchanged."""
    import jax.numpy as jnp

    if not fwhm_mas:
        return image
    n = image.shape[0]
    f = jnp.fft.fftfreq(n, d=pixel_mas / 1000.0 * ARCSEC)  # cycles per radian
    f2 = f[:, None] ** 2 + f[None, :n // 2 + 1] ** 2
    t = jnp.exp(-((jnp.pi * fwhm_mas * ARCSEC / 1000.0) ** 2) * f2 / (4 * jnp.log(2.0)))
    return jnp.fft.irfft2(jnp.fft.rfft2(image) * t, s=(n, n))


def pixels(params, pixel_mas, softmax):
    """The base image (Jy/arcsec^2) of ``params``: ``softplus(base)`` (MPoL), or ``flux softmax(eta) / dx^2`` (``softmax``, virgil's parametrisation: params ``eta`` and ``flux``)."""
    import jax
    import jax.numpy as jnp

    if softmax:
        eta = params["eta"]
        return params["flux"] * jax.nn.softmax(eta.ravel()).reshape(eta.shape) / (pixel_mas / 1000.0) ** 2
    return jax.nn.softplus(params["base"])


def values_of(params, pixel_mas, obs_of_ddid, softmax):
    """virgil values of MPoL-parametrised ``params`` (``log10_amp``: obs 1.., ``log10_sig`` per ddid, ``delta`` (obs 1.., 2) in arcsec): ``log_brightness = ln I``, ``cal.flux = sum I dx^2``, ``cal.gain = amp``,
    ``cal.offset`` in mas and ``cal.sigma = s_M amp`` of the ddid's observation (``alma.values_from_mpol_export``)."""
    import jax.numpy as jnp

    img = pixels(params, pixel_mas, softmax)
    amp = 10.0 ** params["log10_amp"]
    amp_full = jnp.concatenate([jnp.ones(1, amp.dtype), amp])
    return {"log_brightness": jnp.log(jnp.maximum(img, 1e-30)), "cal.flux": jnp.sum(img) * (pixel_mas / 1000.0) ** 2, "cal.gain": amp, "cal.offset": 1000.0 * params["delta"],
            "cal.sigma": 10.0 ** params["log10_sig"] * amp_full[obs_of_ddid]}


def start_params(prob, n_obs, n_ddid, *, softmax=False):
    """MPoL's start (``sgd.py``): the base cube at -3 (or, softmax, eta 0 and the flux ``softplus(-3) dx^2 npix^2``), the log10 amplitudes and sigma factors and the offsets zero."""
    import jax.numpy as jnp

    shape = (prob.npix, prob.npix)
    if softmax:
        flux = float(np.log1p(np.exp(MPOL_RUN["base_start"])) * (prob.pixel_mas / 1000.0) ** 2 * prob.npix**2)
        img = {"eta": jnp.zeros(shape), "flux": jnp.asarray(flux)}
    else:
        img = {"base": jnp.full(shape, MPOL_RUN["base_start"])}
    return {**img, "log10_amp": jnp.zeros(n_obs - 1), "log10_sig": jnp.zeros(n_ddid), "delta": jnp.zeros((n_obs - 1, 2))}


def make_optimizer(lr=MPOL_RUN["lr"], weight_decay=MPOL_RUN["weight_decay"]):
    """``optax.adamw`` as ``torch.optim.AdamW(lr)`` (betas 0.9, 0.999, eps 1e-8, decoupled decay on every parameter; optax's own default decay is 1e-4, torch's 1e-2)."""
    import optax

    return optax.adamw(lr, b1=0.9, b2=0.999, eps=1e-8, weight_decay=weight_decay)


def replica_loss(params, term, *, n_points, obs_of_ddid, pixel_mas, lam_m, blur_mas, entropy_blur_mas, softmax, prior=MPOL["prior"], tot_flux=MPOL["f_tot"]):
    """MPoL's loss of a batch, on all of ``term``'s data: ``-ln L / (2 n) + lambda_M S_M`` (``neg_log_likelihood_avg`` plus ``lam_ent * losses.entropy`` of the blurred image). ``blur_mas`` is in the
    term's nuisance (virgil's ``blur_fwhm_mas``), ``entropy_blur_mas`` is the blur of the entropy's image (``blur_mas`` for MPoL, 0 for the entropy of the base image)."""
    from .alma_gp import term_loss

    vals = values_of(params, pixel_mas, obs_of_ddid, softmax)
    img = pixels(params, pixel_mas, softmax)
    return term_loss(term, vals) / (2.0 * n_points) + lam_m * mpol_entropy(gauss_blur(img, entropy_blur_mas, pixel_mas), prior, tot_flux)


def run_stage(params, term, *, steps, opt, loss_kw, chunk=100, time_limit=None, log=None, to32=True):
    """``steps`` full-batch AdamW steps on ``term`` from ``params`` (a fresh optimiser, as MPoL's second stage). Returns ``(params, losses, wall_s, stopped)``: ``losses`` the per-step loss (numpy),
    run in ``chunk``-step ``lax.scan`` blocks (compiled once per block length); ``time_limit`` (s) stops after a block. Parameters and optimiser state are float32 unless ``to32`` is false."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    if to32:
        params = jax.tree.map(lambda x: jnp.asarray(x, jnp.float32), params)
    state = opt.init(params)

    @eqx.filter_jit
    def block(params, state, term, n):
        def body(carry, _):
            p, s = carry
            loss, g = jax.value_and_grad(lambda q: replica_loss(q, term, **loss_kw))(p)
            g = jax.tree.map(lambda a, b: a.astype(b.dtype), g, p)
            upd, s = opt.update(g, s, p)
            p = optax.apply_updates(p, upd)
            return (p, s), loss

        (p, s), losses = jax.lax.scan(body, (params, state), None, length=n)
        return p, s, losses

    t0, done, out, stopped = time.time(), 0, [], None
    while done < steps:
        n = min(chunk, steps - done)
        params, state, ls = block(params, state, term, n)
        out.append(np.asarray(ls, float))
        done += n
        if log and (done % (10 * chunk) == 0 or done == steps):
            log(f"  replica: {done}/{steps} steps, loss {out[-1][-1]:.8f}, {time.time() - t0:.0f} s")
        if not np.all(np.isfinite(out[-1])):
            stopped = "non_finite"
            break
        if time_limit is not None and time.time() - t0 > time_limit and done < steps:
            stopped = "time_limit"
            break
    return params, np.concatenate(out) if out else np.zeros(0), time.time() - t0, stopped


def obs_of_ddid_of(data):
    """The observation of each ddid (index by ddid) from the data's points."""
    dd, ob = np.asarray(data.ddid), np.asarray(data.obs)
    return np.array([ob[dd == d][0] if np.any(dd == d) else 0 for d in range(int(dd.max()) + 1)])


def replica_compare(base, blurred, mpol_base, mpol_blurred, flux, *, mpol_flux=None, pixel_mas=MPOL["pixel_mas"], ref=REPLICA_PASS):
    """The replica pass (Ben, 2026-10-28) of a replica ``base`` image (Jy/arcsec^2, row 0 North) and its 30 mas ``blurred`` image against MPoL's export (``base_image``, ``image``):

    - Pearson correlation of the two blurred images >= ``ref['correlation']`` (0.99);
    - the base image's N_eff ratio (base over blurred, ``dots_report``) within ``ref['n_eff_ratio_tol']`` (20 %) of ``ref['n_eff_ratio']`` (0.78);
    - the base peak within a factor ``ref['peak_factor']`` (2) of ``ref['peak_base']`` (10 Jy/arcsec^2);
    - the flux within ``ref['flux_tol']`` (5 %) of ``ref['flux']`` (0.294 Jy; ``mpol_flux``, the export's, is recorded beside it).

    MPoL's own numbers from the export (correlation 1, its N_eff ratio, base peak, flux) are returned beside them. Returns ``dict(verdict, items, mpol)``."""
    base, blurred, mb, mt = (np.asarray(x, float) for x in (base, blurred, mpol_base, mpol_blurred))
    corr = float(np.corrcoef(blurred.ravel(), mt.ravel())[0, 1])
    dr = dots_report(base, blurred, pixel_mas)
    dm = dots_report(mb, mt, pixel_mas)
    item = lambda key, label, value, ok, limit: dict(key=key, label=label, value=value, limit=limit, status="PASS" if ok else "FAIL")  # noqa: E731
    items = [
        item("a", "correlation of the blurred images", corr, corr >= ref["correlation"], f">= {ref['correlation']:g}"),
        item("b", f"base N_eff ratio within {ref['n_eff_ratio_tol']:.0%} of {ref['n_eff_ratio']:g}", dr["n_eff_ratio"], abs(dr["n_eff_ratio"] / ref["n_eff_ratio"] - 1) <= ref["n_eff_ratio_tol"],
             f"{ref['n_eff_ratio'] * (1 - ref['n_eff_ratio_tol']):.3g} to {ref['n_eff_ratio'] * (1 + ref['n_eff_ratio_tol']):.3g}"),
        item("c", f"base peak within {ref['peak_factor']:g}x of {ref['peak_base']:g} Jy/arcsec^2", float(base.max()), ref["peak_base"] / ref["peak_factor"] <= base.max() <= ref["peak_base"] * ref["peak_factor"],
             f"{ref['peak_base'] / ref['peak_factor']:.3g} to {ref['peak_base'] * ref['peak_factor']:.3g}"),
        item("d", f"flux within {ref['flux_tol']:.0%} of {ref['flux']:g} Jy", float(flux), abs(flux / ref["flux"] - 1) <= ref["flux_tol"], f"{ref['flux'] * (1 - ref['flux_tol']):.4g} to {ref['flux'] * (1 + ref['flux_tol']):.4g}"),
    ]
    return dict(verdict="PASS" if all(i["status"] == "PASS" for i in items) else "FAIL", items=items,
                mpol=dict(n_eff_ratio=dm["n_eff_ratio"], peak_base=float(mb.max()), flux=mpol_flux, base_dead_in_disk_1e4=dm["base_dead_in_disk_1e4"]),
                replica=dict(n_eff_ratio=dr["n_eff_ratio"], base_dead_in_disk_1e4=dr["base_dead_in_disk_1e4"], sub_taper_power=dr["sub_taper_power"]))


def replica_markdown(res):
    """The replica pass as markdown lines."""
    rows = [f"**Replica pass against MPoL's export: {res['verdict']}** (Ben, 2026-10-28)", "", "| | check | value | limit | |", "|---|---|---|---|---|"]
    rows += [f"| {i['key']} | {i['label']} | {i['value']:.6g} | {i['limit']} | {i['status']} |" for i in res["items"]]
    m = res["mpol"]
    rows += ["", f"MPoL's own export: N_eff ratio {m['n_eff_ratio']:.3f}, base peak {m['peak_base']:.3f} Jy/arcsec^2, flux {m['flux']}, base dead in disk (1e-4) {m['base_dead_in_disk_1e4']:.3%}."]
    return rows


def blurred_of(image, pixel_mas, fwhm_mas=MPOL["taper_mas"]):
    """``tapered_image`` of ``image`` (the 30 mas blur the replica's blurred image is compared at)."""
    return tapered_image(image, pixel_mas, fwhm_mas)


def stage_plan(spec, data, steps=None):
    """``[(name, use, steps)]`` of the replica: ``spec`` a comma list of ``sb`` (obs 0-4) and ``all``; each stage ``steps_for`` its points (``MPOL_RUN``), or, with ``steps`` (the total), shared in
    proportion to those."""
    names = [s.strip() for s in spec.split(",")]
    if not names or any(s not in ("sb", "all") for s in names):
        raise ValueError(f"--replica-stages is a comma list of sb and all, not {spec!r}")
    uses = [(n, data.select(obs=SB_OBS) if n == "sb" else None) for n in names]
    own = [steps_for(len(u.u) if u is not None else len(data.u)) for _, u in uses]
    if steps is not None:
        own = [max(1, int(round(steps * o / sum(own)))) for o in own]
    return [(n, u, k) for (n, u), k in zip(uses, own)]
