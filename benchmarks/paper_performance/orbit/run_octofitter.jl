# Octofitter side of the orbit comparison (plan section 5). One process = one (case, config, thread setting, repeat).
# Octofitter only: this file never imports virgil. Conventions are Octofitter's own (docs/src/rel-astrom.md,
# fit-interfere.md of the installed 9.0.0 package); the mapping to the shared model is fixed by gate 1 (gates.py).
#
#   julia -t 1 --project=julia run_octofitter.jl --case O-A-sim --config matched --draws 200 --warmup 200 --out DIR
#
# Prints one `BENCH_INNER {json}` line (t_calib, t_import, t_load, t_init, t_cold, t_warm, ...) as common.py expects.
# Draws are written as DIR/chain<k>.csv (natural-domain columns of the Octofitter chain).

const T_START = time_ns()
tsec(t0) = (time_ns() - t0) / 1e9

# --- calibration loop: identical to ../calib.jl (copied so the value can be returned and timed in-process) -------------
using LinearAlgebra
function calibrate(n_loop = 2_000_000, n_mat = 300, n_rep = 5)
    t0 = time_ns()
    s = 0.0
    for i in 0:(n_loop - 1)
        s += (i % 7) * 0.5 - (i % 3) * 0.25
    end
    a = reshape(Float64.(0:(n_mat * n_mat - 1)), n_mat, n_mat) .% 11 ./ 11.0
    for _ in 1:n_rep
        a = a * a
        a ./= maximum(abs.(a))
    end
    s += a[1, 1]
    return (time_ns() - t0) / 1e9
end
calibrate(1000, 20, 1)
const T_CALIB = calibrate()

function parse_args(args)
    d = Dict("case" => "O-A-sim", "config" => "matched", "draws" => "1000", "warmup" => "1000", "chains" => "4",
             "seed" => "1", "out" => "", "pull-seeds" => "20", "max-depth" => "10", "no-warm" => "false",
             "corrections" => "off", "pigeons-rounds" => "8", "level-b-sampler" => "nuts")
    i = 1
    while i <= length(args)
        k = lstrip(args[i], '-')
        if k == "no-warm"; d[k] = "true"; i += 1
        else d[k] = args[i + 1]; i += 2 end
    end
    return d
end
const A = parse_args(ARGS)

const T_IMPORT0 = time_ns()
using Octofitter, Distributions
using OctofitterInterferometry
using Pigeons
using Random
using CSV, JSON3, TypedTables
const T_IMPORT = tsec(T_IMPORT0)

const DATA = joinpath(homedir(), ".cache", "virgil-bench", "data", "orbit")
const T_REF_SIM, T_REF_REAL, PLX_SIM, PLX_REAL = 60000.0, 59330.0, 20.0, 16.176
const CASE, CONFIG = A["case"], A["config"]
const LEVEL = CASE == "O-B-sim" ? "B" : "A"
const T_REF = CASE == "O-A-real" ? T_REF_REAL : T_REF_SIM
const PLX = CASE == "O-A-real" ? PLX_REAL : PLX_SIM
# corrections: forced off in `matched` (R7), automatic in `native`
const CORR = CONFIG == "matched" ? false : :auto
const PHASE = CONFIG == "matched" ? :M0 : :θ

period_prior() = CASE == "O-A-real" ?
    truncated(Normal(10.658451, 0.001), 10.658451 - 5 * 0.001, 10.658451 + 5 * 0.001) : Uniform(150.0, 250.0)

function build_system(obs_fn; flux = false)
    A_ = Body(name = "A", variables = @variables begin
        mass ~ Uniform(0.5, 5.0)
        flux_H = 1.0
    end)
    b = if PHASE == :M0
        Body(name = "b", about = A_, variables = @variables begin
            mass = 0.0
            P ~ period_prior()
            e ~ Uniform(0.0, 0.9)
            i ~ Sine()
            ω ~ UniformCircular()
            Ω ~ UniformCircular()
            M0 ~ UniformCircular()
            epoch = $T_REF
            flux_H ~ Uniform(0.01, 1.0)
        end)
    else
        Body(name = "b", about = A_, variables = @variables begin
            mass = 0.0
            P ~ period_prior()
            e ~ Uniform(0.0, 0.9)
            i ~ Sine()
            ω ~ UniformCircular()
            Ω ~ UniformCircular()
            θ ~ UniformCircular()
            epoch = $T_REF
            flux_H ~ Uniform(0.01, 1.0)
        end)
    end
    obs = obs_fn(A_, b)
    sysvars = @variables begin
        plx = $PLX
    end
    return System(name = "bench", bodies = [A_, b], observations = [obs],
                  observing_geometry = CORR, barycentric_lighttime = CORR, variables = sysvars)
end

function level_a(A_, b)
    tab = CSV.read(joinpath(DATA, CASE == "O-A-sim-pulls" ? "astrometry_O-A-sim-pulls_seed00.csv" :
                                  "astrometry_$(CASE).csv"), Table)
    tab = Table(epoch = tab.epoch, ra = tab.ra, dec = tab.dec, σ_ra = tab.sigma_ra, σ_dec = tab.sigma_dec, cor = tab.cor)
    return RelAstromObs(tab; target = b, ref = A_, name = "relastrom")
end

function level_b(A_, b)
    meta = CSV.read(joinpath(DATA, "epochs_O-B-sim.csv"), Table)
    rows = [(; filename = joinpath(DATA, "oifits_O-B-sim", f), epoch = e, use_vis2 = true)
            for (f, e) in zip(meta.file, meta.epoch)]
    return InterferometryObs(Table(rows), targets = (A_, b), ref = A_, band = :H, name = "sim",
                             variables = @variables begin
                                 platescale = 1.0
                                 northangle = 0.0
                                 σ_cp_jitter = 0.0
                             end)
end

function write_chain(dir, k, chain)
    mkpath(dir)
    nm = string.(names(chain))
    M = chain.value.data[:, :, 1]
    open(joinpath(dir, "chain$(k).csv"), "w") do io
        println(io, join(nm, ","))
        for r in 1:size(M, 1)
            println(io, join(M[r, :], ","))
        end
    end
end

function main()
    t_load0 = time_ns()
    sys = build_system(LEVEL == "A" ? level_a : level_b)
    model = Octofitter.LogDensityModel(sys)
    t_load = tsec(t_load0)
    warm = parse(Int, A["warmup"]); draws = parse(Int, A["draws"]); nch = parse(Int, A["chains"])
    seed = parse(Int, A["seed"]); maxd = parse(Int, A["max-depth"])
    use_pigeons = LEVEL == "B" && CONFIG == "native"

    function one_call(seed0)
        chains = []
        for k in 1:nch
            rng = Random.Xoshiro(seed0 * 100 + k)
            if use_pigeons
                res = octofit_pigeons(model; n_rounds = parse(Int, A["pigeons-rounds"]), multithreaded = Threads.nthreads() > 1)
                push!(chains, res.chain)
            else
                push!(chains, octofit(rng, model, 0.8; adaptation = warm, iterations = draws, max_depth = maxd, verbosity = 0))
            end
        end
        return chains
    end

    # initialisation: Octofitter's own route (global optimisation + Pathfinder), no user guesses
    t0 = time_ns()
    init_chain = initialize!(model; verbosity = 0)
    t_init = tsec(t0)

    t0 = time_ns()
    chains = one_call(seed)
    t_cold = tsec(t0)
    t_warm = Float64[]
    if A["no-warm"] != "true"
        t0 = time_ns()
        one_call(seed + 1)
        push!(t_warm, tsec(t0))
    end
    if A["out"] != ""
        for (k, c) in enumerate(chains)
            write_chain(A["out"], k, c)
        end
    end
    flags = (t_warm != Float64[] && t_warm[1] > 0.9 * t_cold) ? ["warm_recompiled"] : String[]
    flags = CORR == false ? flags : vcat(flags, String[])
    out = Dict(
        "t_calib" => T_CALIB, "t_import" => T_IMPORT, "t_load" => t_load, "t_init" => t_init, "t_cold" => t_cold,
        "t_warm" => t_warm, "t_inner_total" => tsec(T_START), "rusage_peak_mb" => Sys.maxrss() / 1e6,
        "thread_env" => Dict("julia_threads" => Threads.nthreads()), "flags" => flags,
        "n_data" => LEVEL == "A" ? 2 * (CASE == "O-A-real" ? 12 : 10) : 1440, "n_param" => model.D,
        "iterations" => Dict("sampling_calls" => 1 + length(t_warm)),
        "accuracy" => Dict(),
        "settings" => Dict("config" => CONFIG, "case" => CASE, "phase_parameter" => string(PHASE), "warmup" => warm,
                           "draws" => draws, "chains" => nch, "chain_method" => "sequential calls", "target_accept" => 0.8,
                           "max_tree_depth" => maxd, "sampler" => use_pigeons ? "octofit_pigeons" : "octofit",
                           "mass_matrix" => "dense (AdvancedHMC DenseEuclideanMetric)",
                           "observing_geometry" => string(CORR), "barycentric_lighttime" => string(CORR),
                           "versions" => Dict("julia" => string(VERSION), "octofitter" => string(pkgversion(Octofitter)))),
        "provenance" => Dict("julia" => string(VERSION)),
    )
    println("BENCH_INNER ", JSON3.write(out))
end

# `include`d by gate_octofitter.jl for the model builders; only run as a script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
