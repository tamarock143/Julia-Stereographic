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


d = 100
sigma = sqrt(d)I(d)
mu = zeros(d)
nu = 100

d > 1 ? x0 = sigma*normalize(randn(d)) + mu : x0 = (sigma*rand([1,-1]) + mu)[1,1]

f = x -> -sum(x.^2)/2

d > 1 ? gradlogx = x -> ForwardDiff.gradient(f,x) : gradlogx = x -> ForwardDiff.derivative(f,x)

#This is here to precalculate the gradient function
gradlogx(x0)



z0 = SPinv(x0; sigma = sigma, mu = mu)
p0 = (I - z0*z0')randn(d+1)

gradlogf = SPgradlog(gradlogx)

h = 1*d^(-3/4)
T = 0.5

(z1,p1) = Rattle(z -> gradlogf(z; sigma=sigma, mu=mu), z0, p0, h, 1)
(z2, p2) = Rattle(z -> gradlogf(z; sigma=sigma, mu=mu), z1, -p1, h, 1)

sum(x -> x^2, z0 - z2)
sum(x -> x^2, p0 + p2)

L = floor(Int64, T/h)

ratpathz = Array{Float64}(undef, L, d+1)
ratpathp = Array{Float64}(undef, L, d+1)

ratpathx = Array{Float64}(undef, L, d)

ratpathz[1,:] = z0; ratpathp[1,:] = p0
ratpathx[1,:] = SP(z0; sigma=sigma, mu=mu)

(z,p) = (z0,p0)

for i in 2:L
    (z,p) = Rattle(z -> gradlogf(z; sigma=sigma, mu=mu), z, p, h, 1)

    ratpathz[i,:] = z; ratpathp[i,:] = p
    ratpathx[i,:] = SP(z; sigma=sigma, mu=mu)
end

plot(ratpathz[:,1], ratpathz[:,2])
plot(ratpathx[:,1], ratpathx[:,2])
