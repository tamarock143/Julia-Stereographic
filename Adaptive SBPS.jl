#Import SBPS Simulator
include("SBPS with Bounce.jl")

times = zeros(10)
beta = 0.5

for k in 1:10
    times[k] = 2^findfirst(map(x -> 2^x, 0:10) .>= k^beta)
end

times