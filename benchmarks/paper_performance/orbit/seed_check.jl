# Behavioural check that Octofitter's initialisation is reproducible with the harness's generator:
#   julia -t 1 --project=julia seed_check.jl --case O-A-sim      (prints SEED_OK, or SEED_DIFFERS and exits 1)
include(joinpath(@__DIR__, "run_octofitter.jl"))
function starts(seed)
    sys = build_system((A_, b) -> level_a(A_, b, 0))
    model = Octofitter.LogDensityModel(sys)
    initialize!(Random.Xoshiro(seed), model; verbosity = 0)
    return [collect(s) for s in model.starting_points]
end
a, b, c = starts(100), starts(100), starts(200)
if a == b && a != c
    println("SEED_OK")
else
    println("SEED_DIFFERS same_seed_equal=", a == b, " other_seed_differs=", a != c)
    exit(1)
end
