# The rel-astrom tutorial of the installed Octofitter 9.0.0 docs (docs/src/rel-astrom.md), code blocks run in order
# (plots omitted) with a short fit. A check that the pinned project works; not a benchmark.
#   julia --project=julia docs_tutorial.jl
using Octofitter, Distributions, Random
A = Body(name = "A", variables = @variables begin
    mass ~ truncated(Normal(1.2, 0.1), lower = 0.1)
end)
planet_b = Body(name = "b", about = A, variables = @variables begin
    mass = 0.0
    a ~ Uniform(0, 100)
    e ~ Uniform(0.0, 0.5)
    i ~ Sine()
    ω ~ UniformCircular()
    Ω ~ UniformCircular()
    θ ~ UniformCircular()
    epoch = 50420.0
end)
astrom_dat_1 = Table(;
    epoch = [50000, 50120, 50240, 50360, 50480, 50600, 50720, 50840],
    ra = [-505.764, -502.57, -498.209, -492.678, -485.977, -478.11, -469.08, -458.896],
    dec = [-66.9298, -37.4722, -7.92755, 21.6356, 51.1472, 80.5359, 109.729, 138.651],
    σ_ra = fill(10.0, 8), σ_dec = fill(10.0, 8), cor = zeros(8))
astrom_obs_1 = RelAstromObs(astrom_dat_1; target = planet_b, ref = A, name = "relastrom")
sys = System(name = "Tutoria", bodies = [A, planet_b], observations = [astrom_obs_1], variables = @variables begin
    plx ~ truncated(Normal(50.0, 0.02), lower = 0.1)
end)
model = Octofitter.LogDensityModel(sys)
init_chain = initialize!(model)
chain = octofit(Random.Xoshiro(1), model; iterations = 200, adaptation = 200, verbosity = 0)
println("TUTORIAL_OK draws=", size(chain, 1), " median a=", median(vec(chain["b_a"])))
