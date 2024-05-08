#Random miscellaneous Tests
include("Adaptive SBPS.jl")
include("Hamiltonian MC.jl")
include("Adaptive SRW.jl")
include("SHMC.jl")
include("Adaptive Slice.jl")

using Plots
using SpecialFunctions
using StatsBase
using JLD

d = 2
sigma = sqrt(d)I(d)
mu = zeros(d) .+ 1e6

nu = 2

d > 1 ? x0 = sigma*normalize(randn(d)) + mu : x0 = (sigma*rand([1,-1]) + mu)[1]

f = x -> -(nu+d)*log(nu+sum(x.^2))

Nslice::Int64 = 500

beta = 1.1
burninslice = 500
adaptlengthslice = 500
R = 1e9
r = 1e-3

@time sliceout = SliceAdaptive(f, x0, Nslice, beta, r, R; sigma, mu, burnin = burninslice, adaptlength = adaptlengthslice);

#@time sliceout = SliceSimulator(f, x0, Nslice; sigma, mu);

#Plot comparison against the true distribution
p(x) = 1/sqrt(2pi)*exp(-x^2/2)
#q(x) = 1/sqrt(2pi*sigmaf)*exp(-x^2/2sigmaf)
q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
b_range = range(-10,10, length=101)

reduce(hcat,[mySSStest() for _ in 1:6])