#Random miscellaneous Tests
include("Adaptive SBPS.jl")
include("Hamiltonian MC.jl")
include("Adaptive SRW.jl")
include("SHMC.jl")
include("Adaptive Slice.jl")

using ForwardDiff
using Plots
using SpecialFunctions
using StatsBase
using JLD

d = 50
sigma = sqrt(d)I(d)
mu = zeros(d) .+ 1e6

nu = 1

d > 1 ? x0 = sigma*normalize(randn(d)) + mu : x0 = (sigma*rand([1,-1]) + mu)[1]

f = x -> -(d+nu)/2*log(nu+sum(x.^2))

d > 1 ? gradlogf = x -> ForwardDiff.gradient(f,x) : gradlogf = x -> ForwardDiff.derivative(f,x)

#This is here to precalculate the gradient function
gradlogf(x0)

Nslice::Int64 = 1e6

beta = 1.1
burnin = 5000
adaptlength = 5000
R = 1e9
r = 1e-3

@time sliceout = SliceAdaptive(f, x0, Nslice, beta, r, R; sigma, mu, burnin, adaptlength);

#@time sliceout = SliceSimulator(f, x0, Nslice; sigma, mu);

#Plot comparison against the true distribution
p(x) = 1/sqrt(2pi)*exp(-x^2/2)
#q(x) = 1/sqrt(2pi*sigmaf)*exp(-x^2/2sigmaf)
q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
b_range = range(-8,8, length=101)

histogram(sliceout.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
plot!(p, label= "N(0,1)", lw=3)
plot!(q, label= "t", lw=3)
xlabel!("x")
ylabel!("P(x)")

plot(sliceout.x[:,1], label = "x1")
vline!(cumsum(sliceout.times[1:end-1]), label = "Adaptations", lw = 0.5)

plot(sliceout.z[:,end], label = "z_{d+1}")
vline!(cumsum(sliceout.times[1:end-1]), label = "Adaptations", lw = 0.5)

plot(sliceout.x[:,1],sliceout.x[:,2])


#map(x -> sum(x -> x^2, x), eachrow(sliceout.mu))
#map(x -> sum(x -> x^2, eigen(x - sqrt(d)I(d)).values), sliceout.sigma)
