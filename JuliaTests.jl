include("Adaptive SBPS.jl")
include("Hamiltonian MC.jl")

using ForwardDiff
using Plots
using SpecialFunctions
using StatsBase

d = 200
sigma = sqrt(d)I(d)
mu = zeros(d)
nu = 3

f = x -> -sum(x.^2)/2

x0 = randn(d)

length(x0) > 1 ? gradlogf = x -> ForwardDiff.gradient(f,x) : gradlogf = x -> ForwardDiff.derivative(f,x)

#This is here to precalculate the gradient function
gradlogf(x0)



### SBPS Testing

T = 10
delta = 0.1
Tbrent = pi/100
tol = 1e-6
lambda = 1

beta = 1.5
burnin = 100
R = 1e12
r = 1e-12

@time out = SBPSAdaptive(gradlogf, x0, lambda, T, delta, beta, r, R; Tbrent = Tbrent, tol = tol,
sigma = sigma, mu = mu, burnin = burnin);

FullSBPS = function ()
    (zout,vout) = SBPSSimulator(gradlogf, x0, lambda, T, delta; Tbrent = Tbrent, tol = tol,
    sigma = sigma, mu = mu);

    n = floor(BigInt, T/delta)+1 #Total number of observations of the skeleton path
    xout = zeros(n,d)

    #Project each entry back to R^d
    for i in 1:n
        xout[i,:] = SP(zout[i,:]; sigma = nu/(nu-2)*sigma, mu = mu)
    end

    return (z = zout, v = vout, x = xout)
end

#@time out = FullSBPS();

#Plot comparison against the true distribution
p(x) = 1/sqrt(2pi)*exp(-x^2/2)
#q(x) = 1/sqrt(2pi*sigmaf)*exp(-x^2/2sigmaf)
#p(x) = 1/2*exp(-abs(x))
q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
b_range = range(-5,5, length=101)

histogram(out.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
#plot!([q p], label= ["t" "N(0,1)"], lw=3)
plot!(q, label= "t", lw=3)
xlabel!("x")
ylabel!("P(x)")

plot(0:delta:T,out.x[:,1], label = "x")
vline!(cumsum(out.times[1:end-1]), label = "Adaptations", lw = 0.5)

map(x -> sum(x -> x^2, x - mu), eachrow(out.mu))
map(x -> sum(x -> x^2, eigen(x - sqrt(d)I(d)).values), out.sigma)

plot(out.x[:,1],out.x[:,2])

xnorms = sum(out.x.^2, dims=2)
plot(0:delta:T,sqrt.(xnorms), label = "||x||")

plot(0:delta:T,out.z[:,end], label = "z_{d+1}")
vline!(cumsum(out.times[1:end-1]), label = "Adaptations")

savefig("NormalAdapt500.pdf")



### HMC Testing

delta = 0.1
L = 5
M = I(d)
N::BigInt = 1e5

@time out = HMC(f, gradlogf, x0, N, delta, L; M = M);

#Plot comparison against the true distribution
p(x) = 1/sqrt(2pi)*exp(-x^2/2)
q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
b_range = range(-8,8, length=101)

histogram(out[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
plot!([q p], label= ["t" "N(0,1)"], lw=3)
xlabel!("x")
ylabel!("P(x)")

plot(1:N,out[:,1], label = "x")

plot(out[:,1],out[:,2])

xnorms = sum(out.^2, dims=2)
plot(1:N,sqrt.(xnorms), label = "||x||")


### Misc Tests

plot(log.(eigen(cov(out.x)).values[90:160] .+ 1e-15))
