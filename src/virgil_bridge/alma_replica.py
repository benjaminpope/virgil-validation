"""MPoL's IM Lup procedure emulated in vv's own optax loop (Track B of ``design/alma_imlup_p3.md``: ``--workflow mpol-replica``), not in virgil core.

What MPoL's ``sgd.py`` does (mpol 0.3.1, the p3a run: ``--lr 1e-2 --FWHM 0.03 --lam-ent 1e-9 --epochs 10``, two chained stages, ``train.sbatch``):

- pixels ``I = softplus(base_cube)`` (``BaseCube``), the base cube starting at -3 (1.2837 Jy over the grid);
- the 30 mas Gaussian ``GaussConvFourier`` is a layer of the forward model, and the entropy ``losses.entropy`` (prior_intensity 1e-4, tot_flux 0.253) is of the blurred ``icube``;
- ``log10`` amplitude and sigma factors and the offsets in arcsec are the other parameters (zeros at the start), all under ``torch.optim.AdamW(lr=1e-2)`` whose weight decay is
  torch's default 1e-2 (decoupled, on every parameter, so the factors are pulled to 1 and the offsets to 0);
- the loss of a batch is ``-ln L_b / (2 n_b) + lambda_M S_M`` (``neg_log_likelihood_avg``), batches of 40,000 points, 10 epochs per stage, about 1,061 batches per epoch on all the data.

Here the same loss is taken on all the data at once (full batch, virgil's Toeplitz chi^2, the blur through virgil's ``blur_fwhm_mas``), for the same number of steps, with optax's AdamW at weight
decay 1e-2 on the same parametrisation; the parameters and the optimiser state are float32. The minibatch noise is the one difference by design; ``run_stage_minibatch`` (``--replica-minibatch``) removes it, with MPoL's shuffled 40,000-point batches and per-epoch extra step. Ablations: no weight decay, the softmax
parametrisation of virgil in place of softplus, another step count.
"""

import math
import time

import numpy as np

from .alma import MPOL, ARCSEC, SB_OBS, dots_report, image_of, tapered_image
from .alma_start import CRITERIA

MPOL_RUN = dict(lr=1e-2, weight_decay=1e-2, batch_size=40000, epochs=10, stages=("all", "all"), base_start=-3.0, lam_m=1e-9)  # p3a's flags (train.sbatch); both of its stages ran on all the data
MPOL_STEPS = dict(sb=10 * 273, all=10 * 1061)  # IM Lup: epochs x ceil(N / batch_size), N = 10,883,819 (obs 0-4) and 42,421,131 (p3a summary.md)
REPLICA_PASS = CRITERIA["replica"]  # Ben's pass, 2026-10-28: hashed with the rest of the criteria


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


def run_stage(params, term, *, steps, opt, loss_kw, chunk=100, time_limit=None, log=None, to32=True, stats=None):
    """``steps`` full-batch AdamW steps on ``term`` from ``params`` (a fresh optimiser, as MPoL's second stage). Returns ``(params, losses, wall_s, stopped)``: ``losses`` the per-step loss (numpy),
    run in ``chunk``-step ``lax.scan`` blocks (compiled once per block length); ``time_limit`` (s) stops after a block. Parameters and optimiser state are float32 unless ``to32`` is false.
    ``stats`` (a dict) receives ``peak_offset_mas``: the largest |offset| (mas) the optimiser visited, which the Toeplitz term's offset box has to contain (MPoL's offsets are unbounded)."""
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
            return (p, s), (loss, 1000.0 * jnp.max(jnp.abs(p["delta"]), initial=0.0))

        (p, s), (losses, peaks) = jax.lax.scan(body, (params, state), None, length=n)
        return p, s, losses, jnp.max(peaks)

    t0, done, out, stopped, peak = time.time(), 0, [], None, 0.0
    while done < steps:
        n = min(chunk, steps - done)
        try:
            params, state, ls, pk = block(params, state, term, n)
        except eqx.EquinoxRuntimeError:  # an offset left the Toeplitz box inside this block: say how far the completed blocks had gone, then fail as before
            if log:
                log(f"  replica: offset left the Toeplitz box in steps {done + 1}-{done + n}; peak |offset| over the {done} completed steps was {peak:.1f} mas")
            raise
        out.append(np.asarray(ls, float))
        peak = max(peak, float(pk))
        if stats is not None:
            stats["peak_offset_mas"] = peak
        done += n
        if log and (done % (10 * chunk) == 0 or done == steps):
            log(f"  replica: {done}/{steps} steps, loss {out[-1][-1]:.8f}, peak |offset| {peak:.1f} mas, {time.time() - t0:.0f} s")
        if not np.all(np.isfinite(out[-1])):
            stopped = "non_finite"
            break
        if time_limit is not None and time.time() - t0 > time_limit and done < steps:
            stopped = "time_limit"
            break
    return params, np.concatenate(out) if out else np.zeros(0), time.time() - t0, stopped


def epoch_batches(n, batch=MPOL_RUN["batch_size"], seed=0, epoch=0, stage=0):
    """The index batches of one epoch as ``DataLoader(shuffle=True, drop_last=False)`` makes them: a seeded permutation of ``range(n)`` (a function of ``seed``, ``stage`` and ``epoch``: the stages do not repeat each other's order) cut into ``batch`` points, the last
    batch partial. Every point is in exactly one batch; there are ``ceil(n / batch)`` of them."""
    perm = np.random.default_rng([int(seed), int(stage), int(epoch)]).permutation(n)
    return [perm[i:i + batch] for i in range(0, n, batch)]


def take(data, idx):
    """The points ``idx`` (any order) of a ``ComplexVisData``, labels unchanged (``ComplexVisData.subset`` with an index in place of a mask, which would cost O(N) per batch)."""
    import dataclasses

    keep = ("ddid_ids", "obs_ids", "ddid_obs")
    return type(data)(**{f.name: (getattr(data, f.name) if f.name in keep or getattr(data, f.name) is None else getattr(data, f.name)[idx]) for f in dataclasses.fields(data)})


def run_stage_minibatch(params, data, make_term, *, steps, opt, loss_kw, batch=MPOL_RUN["batch_size"], seed=0, stage=0, time_limit=None, log=None, to32=True, stats=None, log_every=1000):
    """``steps`` minibatch AdamW steps as MPoL's ``train`` loop (``sgd.py``): per epoch the points are reshuffled (``epoch_batches``) and each batch takes one step on its own loss ``-ln L_b / (2 n_b) + lambda_M S_M``
    (``replica_loss`` with ``n_points`` the batch's size, the last batch of an epoch partial), the data term ``make_term(batch_data)`` the direct (non-Toeplitz) term of that batch. After each full epoch one more
    ``optimizer.step()`` is taken with the last batch's gradient, as ``sgd.py`` does (its ``zero_grad`` is at the top of the batch loop, so the gradient is still there): it is not counted in ``steps`` and has
    no loss. The optimiser is fresh (a second stage starts a new one). Returns ``(params, losses, wall_s, stopped)`` like ``run_stage``; ``losses`` has one entry per batch step (each the loss of one batch: its points and its ``n_b``, not the stage's objective). ``stats`` also receives ``epoch_loss`` (the ``n_b``-weighted mean of each epoch's batch losses, the last perhaps partial) and ``epoch_steps`` (the steps done at each epoch's end)."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    if to32:
        params = jax.tree.map(lambda x: jnp.asarray(x, jnp.float32), params)
    state = opt.init(params)

    @eqx.filter_jit
    def step(params, state, term):
        loss, g = jax.value_and_grad(lambda q: replica_loss(q, term, **{**loss_kw, "n_points": term.n}))(params)  # term.n is static: the batch's own size
        g = jax.tree.map(lambda a, b: a.astype(b.dtype), g, params)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, loss, g

    @eqx.filter_jit
    def extra(params, state, g):
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state

    t0, done, out, stopped, peak, epoch, n = time.time(), 0, [], None, 0.0, 0, len(data.u)
    ep_loss, ep_steps = [], []
    while done < steps and stopped is None:
        g, wsum = None, 0.0
        batches = epoch_batches(n, batch, seed, epoch, stage)
        for i, idx in enumerate(batches):
            params, state, loss, g = step(params, state, make_term(take(data, idx)))
            out.append(float(loss))
            wsum += out[-1] * len(idx)
            done += 1
            peak = max(peak, 1000.0 * float(jnp.max(jnp.abs(params["delta"]), initial=0.0)))
            if stats is not None:
                stats["peak_offset_mas"] = peak
            if log and done % log_every == 0:
                log(f"  replica: {done}/{steps} minibatch steps (epoch {epoch + 1}), loss {out[-1]:.8f}, peak |offset| {peak:.1f} mas, {time.time() - t0:.0f} s")
            if not np.isfinite(out[-1]):
                stopped = "non_finite"
            elif time_limit is not None and time.time() - t0 > time_limit and done < steps:
                stopped = "time_limit"
            if stopped or (done >= steps and i < len(batches) - 1):  # a stop, or the steps used up inside an epoch (no extra step then; at an epoch's end they get it)
                break
        else:  # the epoch ran to its end: MPoL's extra optimizer.step() with the last batch's gradient
            params, state = extra(params, state, g)
        ep_loss.append(wsum / sum(len(x) for x in batches[:i + 1]))
        ep_steps.append(done)
        epoch += 1
    if stats is not None:
        stats.update(epoch_loss=ep_loss, epoch_steps=ep_steps)
    return params, np.asarray(out, float), time.time() - t0, stopped


def obs_of_ddid_of(data):
    """The observation of each ddid (index by ddid) from the data's points."""
    dd, ob = np.asarray(data.ddid), np.asarray(data.obs)
    return np.array([ob[dd == d][0] if np.any(dd == d) else 0 for d in range(int(dd.max()) + 1)])


def replica_compare(base, blurred, mpol_base, mpol_blurred, flux, *, mpol_flux=None, pixel_mas=MPOL["pixel_mas"]):
    """The replica pass (Ben, 2026-10-28) of a replica ``base`` image (Jy/arcsec^2, row 0 North) and its 30 mas ``blurred`` image against MPoL's export (``base_image``, ``image``):

    - Pearson correlation of the two blurred images >= ``ref['correlation']`` (0.99);
    - the base image's N_eff ratio (base over blurred, ``dots_report``) within ``ref['n_eff_ratio_tol']`` (20 %) of ``ref['n_eff_ratio']`` (0.78);
    - the base peak within a factor ``ref['peak_factor']`` (2) of ``ref['peak_base']`` (10 Jy/arcsec^2);
    - the flux within ``ref['flux_tol']`` (5 %) of ``ref['flux']`` (0.294 Jy: the flux of MPoL's p3a base image, ``sum(base_image) dx^2`` of its export, not the 0.253 Jy of its entropy's ``tot_flux``; ``mpol_flux``, the export's own, is recorded beside it).

    (b)-(d) are judged against these constants (``CRITERIA['replica']``, hashed; not overridable), which are the p3a export's numbers. The export passed in is checked against the same bands (``mpol['export_in_band']``): when it is another MPoL run (another lambda, more epochs) the bands do not describe it, and the result carries ``warning`` and the markdown says so, so the verdict is not read as agreement with that run.

    MPoL's own numbers from the export (correlation 1, its N_eff ratio, base peak, flux) are returned beside them. Returns ``dict(verdict, items, mpol)``."""
    ref = REPLICA_PASS
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
    band = dict(n_eff_ratio=bool(abs(dm["n_eff_ratio"] / ref["n_eff_ratio"] - 1) <= ref["n_eff_ratio_tol"]), peak_base=bool(ref["peak_base"] / ref["peak_factor"] <= mb.max() <= ref["peak_base"] * ref["peak_factor"]),
                flux=None if mpol_flux is None else bool(abs(mpol_flux / ref["flux"] - 1) <= ref["flux_tol"]))
    in_band = all(v is not False for v in band.values())
    warning = None if in_band else f"MPoL's export is outside the replica bands ({[k for k, v in band.items() if v is False]}): (b)-(d) judge the replica against the p3a constants, not against this export"
    return dict(verdict="PASS" if all(i["status"] == "PASS" for i in items) else "FAIL", items=items, warning=warning,
                mpol=dict(n_eff_ratio=dm["n_eff_ratio"], peak_base=float(mb.max()), flux=mpol_flux, base_dead_in_disk_1e4=dm["base_dead_in_disk_1e4"], export_in_band=band),
                replica=dict(n_eff_ratio=dr["n_eff_ratio"], base_dead_in_disk_1e4=dr["base_dead_in_disk_1e4"], sub_taper_power=dr["sub_taper_power"]))


def replica_markdown(res):
    """The replica pass as markdown lines."""
    rows = [f"**Replica pass against MPoL's export: {res['verdict']}** (Ben, 2026-10-28)", "", "| | check | value | limit | |", "|---|---|---|---|---|"]
    rows += [f"| {i['key']} | {i['label']} | {i['value']:.6g} | {i['limit']} | {i['status']} |" for i in res["items"]]
    m = res["mpol"]
    rows += ["", f"MPoL's own export: N_eff ratio {m['n_eff_ratio']:.3f}, base peak {m['peak_base']:.3f} Jy/arcsec^2, flux {m['flux']}, base dead in disk (1e-4) {m['base_dead_in_disk_1e4']:.3%}."]
    if res.get("warning"):
        rows += ["", f"**Warning:** {res['warning']}."]
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
