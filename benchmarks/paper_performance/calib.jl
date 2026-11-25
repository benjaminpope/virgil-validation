# Fixed calibration loop (about 1 s), the Julia counterpart of calib.py: scalar loop plus a 300 x 300 Float64 matmul.
# Run: `julia -t 1 calib.jl` prints {"t_calib": seconds}. The first call includes compilation, so it is run twice and
# the second timing is reported (compilation is measured separately as t_cold in the benchmark rows).
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
    return (time_ns() - t0) / 1e9, s
end

calibrate(1000, 20, 1)  # warm-up compile
t, s = calibrate()
println("{\"t_calib\": ", t, ", \"checksum\": ", s, "}")
