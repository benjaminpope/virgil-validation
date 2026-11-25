# Octofitter side of gates 1-3 (see gates.py). Reads DATA/gates/gate_in.json, writes DATA/gates/octofitter_out.json.
#   julia -t 1 --project=julia gate_octofitter.jl
# Octofitter only; never imports virgil. Angles are given to Octofitter in radians.
include(joinpath(@__DIR__, "run_octofitter.jl"))   # builders only (main() is guarded); ARGS must not carry options here

const GDIR = joinpath(DATA, "gates")
gin = JSON3.read(read(joinpath(GDIR, "gate_in.json"), String))


function overrides(p; flux = nothing)
    ω, Ω, M = deg2rad(p.omega), deg2rad(p.Omega), deg2rad(p.M0)
    b = (; P = p.P, e = p.e, i = deg2rad(p.inc), ωx = cos(ω), ωy = sin(ω), Ωx = cos(Ω), Ωy = sin(Ω),
         M0x = cos(M), M0y = sin(M))
    flux === nothing || (b = merge(b, (; flux_H = flux)))
    return (; bodies = (; A = (; mass = p.M_tot), b = b))
end

out = Dict{String,Any}("mapping" => "Octofitter ω, Ω, M0 given directly in radians; no sign or offset applied")

# --- gate 1: noise-free ephemeris vs the crosscheck orbit ---------------------------------------------------------
try
    tab = Table(epoch = collect(Float64.(gin.epochs)), ra = zeros(20), dec = zeros(20), σ_ra = fill(0.05, 20),
                σ_dec = fill(0.05, 20), cor = zeros(20))
    pos = Any[]
    for p in gin.elements
        sys = build_system((A_, b) -> RelAstromObs(tab; target = b, ref = A_, name = "g1"))
        θ = Octofitter.drawfrompriors(sys; overrides = overrides(p))
        sim = Octofitter.generate_from_params(sys, θ; add_noise = false)
        o = sim.observations[1].table
        push!(pos, Dict("ra" => collect(Float64.(o.ra)), "dec" => collect(Float64.(o.dec))))
    end
    out["positions"] = pos
catch err
    out["gate1_error"] = sprint(showerror, err, catch_backtrace())
end

# --- gate 2: Level A log-likelihood at two points -------------------------------------------------------------------
try
    sysA = build_system(level_a)
    ll = Float64[]
    for p in gin.point_a
        θ = Octofitter.drawfrompriors(sysA; overrides = overrides(p))
        push!(ll, Octofitter.make_ln_like(sysA)(sysA, θ))
    end
    out["loglike_a"] = ll
catch err
    out["gate2_error"] = sprint(showerror, err, catch_backtrace())
end

# --- gate 3: Level B log-likelihood at two points ---------------------------------------------------------------------
try
    sysB = build_system(level_b)
    ll = Float64[]
    for p in gin.point_b
        θ = Octofitter.drawfrompriors(sysB; overrides = overrides(p; flux = p.flux))
        push!(ll, Octofitter.make_ln_like(sysB)(sysB, θ))
    end
    out["loglike_b"] = ll
catch err
    out["gate3_error"] = sprint(showerror, err, catch_backtrace())
end


# --- gate 4: what the OIFITS reader stored, per epoch (compared with astropy in gates.py) ------------------------------
try
    sysB = build_system(level_b)
    tabB = sysB.observations[1].table
    rows(m) = [collect(Float64.(m[r, :])) for r in 1:size(m, 1)]
    rd = Any[]
    for k in 1:length(tabB.epoch)
        wl = collect(Float64.(tabB.eff_wave[k]))
        push!(rd, Dict("epoch" => tabB.epoch[k], "wl" => wl,
            "u_m" => rows(tabB.u[k] .* wl'), "v_m" => rows(tabB.v[k] .* wl'),
            "vis2" => rows(tabB.vis2_data[k]), "dvis2" => rows(tabB.dvis2[k]),
            "cp_deg" => rows(tabB.cps_data[k]), "dcp_deg" => rows(tabB.dcps[k]),
            "idx1" => collect(Int.(tabB.index_cps1[k])), "idx2" => collect(Int.(tabB.index_cps2[k])),
            "idx3" => collect(Int.(tabB.index_cps3[k]))))
    end
    out["reader_raw"] = rd
catch err
    out["gate4_error"] = sprint(showerror, err, catch_backtrace())
end

open(joinpath(GDIR, "octofitter_out.json"), "w") do io
    JSON3.write(io, out)
end
println("wrote ", joinpath(GDIR, "octofitter_out.json"))
