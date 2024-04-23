#Random miscellaneous Tests
include("Adaptive SBPS.jl")
include("Hamiltonian MC.jl")
include("Adaptive SRW.jl")
include("SHMC.jl")

using ForwardDiff
using Plots
using SpecialFunctions
using StatsBase
using JLD

N::Int64 = 10000

@time out = SSSSimulator(f, x0, N; sigma, mu);

#Plot comparison against the true distribution
p(x) = 1/sqrt(2pi)*exp(-x^2/2)
#q(x) = 1/sqrt(2pi*sigmaf)*exp(-x^2/2sigmaf)
#q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
b_range = range(-5,5, length=101)

histogram(out.x[:,100], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
plot!(p, label= "N(0,1)", lw=3)
#plot!(q, label= "t", lw=3)
xlabel!("x")
ylabel!("P(x)")

plot(out.x[:,1], label = "x1")

plot(out.z[:,end], label = "z_{d+1}")

plot(out.x[:,1],out.x[:,2]) 