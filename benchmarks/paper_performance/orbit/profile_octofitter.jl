# Profiling arm, Octofitter side: the per-call cost of the log density and its gradient, and the cost per leapfrog step of a
# warm octofit call, for the same models as run_octofitter.jl (PR 275). Compare with profile_virgil.py.
#   julia -t 1 --project=julia profile_octofitter.jl --case O-A-sim --config matched --warmup 1000 --draws 1000 --calls 500 --out DIR
# Prints one `PROFILE_OCTOFITTER {json}` line; with --julia-profile 1 also writes DIR/julia_profile.txt (Profile.print, flat, of the
# gradient loop). Octofitter only.
include(joinpath(@__DIR__, "run_octofitter.jl"))   # builders (main() is guarded); the options below are read from ARGS by it
using Statistics, Profile
const LogDensityProblems = Octofitter.LogDensityProblems     # Octofitter's own dependency, not a direct one of this project

const NCALLS = parse(Int, get(A, "calls", "500"))
const OUT = get(A, "out", "")

function timeit(f, n)
    f(); f()                                  # compile
    ts = Float64[]
    for _ in 1:n
        t0 = time_ns(); f(); push!(ts, (time_ns() - t0) / 1e3)      # microseconds
    end
    return Dict("median_us" => median(ts), "min_us" => minimum(ts))
end

function profile_main()
    ds = 0                                     # O-A-sim-pulls: seed 00, as in profile_virgil.py
    sys = build_system(LEVEL == "A" ? (A_, b) -> level_a(A_, b, ds) : level_b)
    model = Octofitter.LogDensityModel(sys)
    initialize!(model; verbosity = 0)
    D = model.D
    rng = Random.Xoshiro(1)
    # a point in the sampler's own (unconstrained) space: the optimiser's start if it is finite, else a small random one
    θ = try
        s = first(model.starting_points)
        length(s) == D && isfinite(LogDensityProblems.logdensity(model, collect(Float64, s))) ? collect(Float64, s) : 0.1 .* randn(rng, D)
    catch
        0.1 .* randn(rng, D)
    end
    per_call = Dict(
        "logdensity" => timeit(() -> LogDensityProblems.logdensity(model, θ), NCALLS),
        "grad" => timeit(() -> LogDensityProblems.logdensity_and_gradient(model, θ), NCALLS))
    if A["julia-profile"] == "1"
        Profile.clear()
        @profile for _ in 1:max(NCALLS, 2000)
            LogDensityProblems.logdensity_and_gradient(model, θ)
        end
        OUT == "" || open(joinpath(OUT, "julia_profile.txt"), "w") do io
            Profile.print(IOContext(io, :displaysize => (200, 200)); format = :flat, sortedby = :count, mincount = 5, noisefloor = 2.0)
        end
    end
    # warm octofit: one call per chain, as run_octofitter.jl does; steps from the chain internals when present
    warm = parse(Int, A["warmup"]); draws = parse(Int, A["draws"]); nch = parse(Int, A["chains"]); maxd = parse(Int, A["max-depth"])
    one(seed0) = [octofit(Random.Xoshiro(seed0 * 100 + k), model, 0.8; adaptation = warm, iterations = draws, max_depth = maxd, verbosity = 0) for k in 1:nch]
    one(1)                                      # compile / first call
    t0 = time_ns(); chains = one(2); t_warm = (time_ns() - t0) / 1e9
    steps = try
        sum(sum(Float64.(vec(chain[:n_steps]))) for chain in chains)
    catch
        NaN
    end
    return Dict("case" => CASE, "config" => CONFIG, "n_param" => D, "chains" => nch, "warmup" => warm, "draws" => draws, "calls" => NCALLS,
                "julia_threads" => Threads.nthreads(), "versions" => Dict("julia" => string(VERSION), "octofitter" => string(pkgversion(Octofitter))),
                "per_call_us" => per_call,
                "warm" => Dict("t_s" => t_warm, "steps_all_chains" => steps, "us_per_step_all" => t_warm * 1e6 / steps,
                               "note" => "chains run one after another; the warm time covers warmup + draws, steps the draws only (NaN: n_steps absent from the chain)"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    out = profile_main()
    println("PROFILE_OCTOFITTER ", JSON3.write(out))
end
