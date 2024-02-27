include("Adaptive SBPS.jl")
include("Hamiltonian MC.jl")

using ForwardDiff
using Plots
using SpecialFunctions
using StatsBase
using JLD

d = 2
sigma = sqrt(d)I(d)
mu = zeros(d)
nu = 1.6

f = x -> log(1+ sum(x.^2)/nu)*-((nu+d)/2)

d > 1 ? x0 = sigma*normalize(randn(d)) + mu : x0 = sigma*rand([1,-1]) + mu

d > 1 ? gradlogf = x -> ForwardDiff.gradient(f,x) : gradlogf = x -> ForwardDiff.derivative(f,x)

#This is here to precalculate the gradient function
gradlogf(x0)


### SBPS Testing

T = 4e7
delta = 0.1
Tbrent = pi/10
Epsbrent = 0.01
tol = 1e-6
lambda = 1

beta = 1.1
burnin = 50
adaptlength = 50
R = 1e9
r = 1e-3

@time out = SBPSAdaptive(gradlogf, x0, lambda, T, delta, beta, r, R; Tbrent, Epsbrent, tol, sigma, mu, burnin, adaptlength);

FullSBPS = function()
    (zout,vout) = SBPSSimulator(gradlogf, x0, lambda, T, delta; Tbrent = Tbrent, Epsbrent = Epsbrent, tol = tol,
    sigma = sigma, mu = mu);

    n = floor(BigInt, T/delta)+1 #Total number of observations of the skeleton path
    xout = zeros(n,d)

    #Project each entry back to R^d
    for i in 1:n
        xout[i,:] = SP(zout[i,:]; sigma = sigma, mu = mu)
    end

    return (z = zout, v = vout, x = xout)
end

#@time out = FullSBPS();

#Plot comparison against the true distribution
p(x) = 1/sqrt(2pi)*exp(-x^2/2)
#q(x) = 1/sqrt(2pi*sigmaf)*exp(-x^2/2sigmaf)
q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
b_range = range(-10,10, length=101)

histogram(out.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
plot!(p, label= "N(0,1)", lw=3)
plot!(q, label= "t", lw=3)
xlabel!("x")
ylabel!("P(x)")

plot(0:delta:T,out.x[:,1], label = "x1")
vline!(cumsum(out.times[1:end-1]), label = "Adaptations", lw = 0.5)
plot!(0:delta:T,out.x[:,2], label = "x2")

#map(x -> sum(x -> x^2, x - mu), eachrow(out.mu))
#map(x -> sum(x -> x^2, eigen(x - sqrt(d)I(d)).values), out.sigma)

#plot(out.x[:,1],out.x[:,2])

xnorms = vec(sum(out.x.^2, dims=2))
#plot(0:delta:T,sqrt.(xnorms), label = "||x||")
#vline!(cumsum(out.times[1:end-1]), label = "Adaptations")

#Comparison of norms with F-distribution
a = 1e5
b = 1e6
norms_range = range(a,b, length = 101)
histogram(xnorms/d, label="||x||", bins=norms_range, normalize=:pdf, color=:gray)
p(x) = 1/beta(d/2,nu/2)*(d/nu)^(d/2)*x^(d/2 - 1)*(1+d/nu*x)^(-(d+nu)/2)
plot!(norms_range, x -> p(x)/(beta_inc(d/2,nu/2,d*b/(d*b+nu))[1] -beta_inc(d/2,nu/2,d*a/(d*a+nu))[1]), label = "F", lw = 3)

plot(autocor(xnorms))

plot(0:delta:T,out.z[:,end], label = "z_{d+1}")
vline!(cumsum(out.times[1:end-1]), label = "Adaptations")

mean(out.z[:,end])

#savefig("tAdaptationsLatitude.pdf")


### HMC Testing

hmcdelta = 1.45d^(-1/4)
L = 5
d > 1 ? M = I(d) : M = 1
N::Int64 = 8e8

@time hmcout = HMC(f, gradlogf, x0, N, hmcdelta, L; M = M);
hmcout.a
#Plot comparison against the true distribution
p(x) = 1/sqrt(2pi)*exp(-x^2/2)
q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
b_range = range(-8,8, length=101)

histogram(hmcout.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
plot!([q p], label= ["t" "N(0,1)"], lw=3)
xlabel!("x")
ylabel!("P(x)")

#plot(hmcout.x[collect(Int64,1:1e0:N),1], label = "x")

#plot(hmcout.x[:,1],hmcout.x[:,2])

hmcxnorms = vec(sum(hmcout.x .^2, dims=2))
#plot(sqrt.(hmcxnorms), label = "||x||")
maximum(hmcxnorms)

### Misc Tests

Z(a) = beta_inc(d/2,nu/2,d*a/(d*a+nu))[2]

plot(10 .^(-2:0.1:11), a -> log(abs(sum(xnorms/d .>= a)/length(xnorms) - Z(a))/Z(a)), label = "SBPS")
plot!(xscale=:log10, minorgrid=true)
plot!(10 .^(-2:0.1:11), a -> log(abs(sum(hmcxnorms/d .>= a)/length(hmcxnorms) - Z(a))/Z(a)), label = "HMC")
title!("Absolute Relative Error for CCDF of norm of a t-distribution\nwith d = 2,  ν = 1.6 (Runtime of ~1000 seconds)", titlefontsize = 10)


mytest = function ()
    @time out = SBPSAdaptive(gradlogf, x0, lambda, T, delta, beta, r, R; Tbrent, Epsbrent, tol, sigma, mu, burnin, adaptlength);

    save("sbps.jld", "SBPS", out)
    xnorms = vec(sum(out.x.^2, dims=2))

    @time hmcout = HMC(f, gradlogf, x0, N, hmcdelta, L; M = M)

    save("hmc.jld", "HMC", hmcout)
    hmcxnorms = vec(sum(hmcout.x .^2, dims=2))

    p = plot(10 .^(-2:0.1:11), a -> log(abs(sum(xnorms/d .>= a)/length(xnorms) - Z(a))/Z(a)), label = "SBPS")
    plot!(p, xscale=:log10, minorgrid=true)
    plot!(p, 10 .^(-2:0.1:11), a -> log(abs(sum(hmcxnorms/d .>= a)/length(hmcxnorms) - Z(a))/Z(a)), label = "HMC")
    title!(p, "Absolute Relative Error for CCDF of norm of a t-distribution\nwith d = 2,  ν = 1.6 (Runtime of ~40000 seconds)", titlefontsize = 10)

    savefig("tNormDistComparison.pdf")
end

mytest()

out = load("sbps.jld")["SBPS"]
hmcout = load("hmc.jld")["HMC"]


### SRW Tests

h2 = 0.1*d^-1
Nsrw::Int64 = 1e6

@time srwout = SRWSimulator(f, x0, h2, Nsrw; sigma = sigma, mu = mu);
srwout.a

p(x) = 1/sqrt(2pi)*exp(-x^2/2)
q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
b_range = range(-8,8, length=101)

histogram(srwout.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
plot!([q p], label= ["t" "N(0,1)"], lw=3)
xlabel!("x")
ylabel!("P(x)")

plot(srwout.z[:,d+1])

#plot(srwout.x[:,1],srwout.x[:,2])

srwxnorms = vec(sum(srwout.x .^2, dims=2))
#plot(sqrt.(srwxnorms), label = "||x||")
maximum(srwxnorms)

plot(10 .^(-2:0.1:11), a -> log(abs(sum(srwxnorms/d .>= a)/length(srwxnorms) - Z(a))/Z(a)), label = "SRW")
plot!(xscale=:log10, minorgrid=true)
