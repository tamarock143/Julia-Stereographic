include("Adaptive SBPS.jl")
include("Hamiltonian MC.jl")

using ForwardDiff
using Plots
using SpecialFunctions
using StatsBase

d = 200
sigma = sqrt(d)I(d)
mu = zeros(d) .+ 1e4
nu = 200

f = x -> -(nu+d)/2*log(nu + sum(x.^2))

d > 1 ? x0 = randn(d) .+ 1e4 : x0 = randn()

d > 1 ? gradlogf = x -> ForwardDiff.gradient(f,x) : gradlogf = x -> ForwardDiff.derivative(f,x)

#This is here to precalculate the gradient function
gradlogf(x0)


### SBPS Testing

T = 1000
delta = 0.1
Tbrent = pi/100
tol = 1e-6
lambda = 1

beta = 1.1
burnin = 100
adaptlength = 20
R = 1e9
r = 1e-6

@time out = SBPSAdaptive(gradlogf, x0, lambda, T, delta, beta, r, R; Tbrent, tol, sigma, mu, burnin, adaptlength);

FullSBPS = function ()
    (zout,vout) = SBPSSimulator(gradlogf, x0, lambda, T, delta; Tbrent = Tbrent, tol = tol,
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
#p(x) = 1/sqrt(2pi)*exp(-x^2/2)
#q(x) = 1/sqrt(2pi*sigmaf)*exp(-x^2/2sigmaf)
q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
b_range = range(-8,8, length=101)

histogram(out.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
#plot!(p, label= "N(0,1)", lw=3)
plot!(q, label= "t", lw=3)
xlabel!("x")
ylabel!("P(x)")

plot(0:delta:T,out.x[:,1], label = "x")
vline!(cumsum(out.times[1:end-1]), label = "Adaptations", lw = 0.5)

map(x -> sum(x -> x^2, x - mu), eachrow(out.mu))
map(x -> sum(x -> x^2, eigen(x - sqrt(d)I(d)).values), out.sigma)

plot(out.x[:,1],out.x[:,2])

plot(autocor(out.x[:,1]))

xnorms = sum(out.x.^2, dims=2)
plot(0:delta:T,sqrt.(xnorms), label = "||x||")
vline!(cumsum(out.times[1:end-1]), label = "Adaptations")

a = 0
b = 50
norms_range = range(a,b, length = 101)
histogram(xnorms/d, label="||x||", bins=norms_range, normalize=:pdf, color=:gray)
p(x) = 1/beta(d/2,nu/2)*(d/nu)^(d/2)*x^(d/2 - 1)*(1+d/nu*x)^(-(d+nu)/2)
plot!(norms_range, x -> p(x)/(beta_inc(d/2,nu/2,d*b/(d*b+nu))[1] -beta_inc(d/2,nu/2,d*a/(d*a+nu))[1]),
    label = "F", lw = 3)

beta_inc(d/2,nu/2,d*a/(d*a+nu))[2]


plot(autocor(xnorms))

plot(0:delta:T,out.z[:,end], label = "z_{d+1}")
vline!(cumsum(out.times[1:end-1]), label = "Adaptations")

#savefig("NormalAdapt500.pdf")


### HMC Testing

delta = 3.02*d^(-1/4)
L = 5
d > 1 ? M = I(d) : M = 1
N::BigInt = 1e6

@time out = HMC(f, gradlogf, x0, N, delta, L; M = M);
out.a
#Plot comparison against the true distribution
p(x) = 1/sqrt(2pi)*exp(-x^2/2)
q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
b_range = range(-8,8, length=101)

histogram(out.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
plot!([q p], label= ["t" "N(0,1)"], lw=3)
xlabel!("x")
ylabel!("P(x)")

plot(1:N,out.x[:,1], label = "x")

plot(autocor(out.x[:,1]))

plot(out.x[:,1],out.x[:,2])

xnorms = sum(out.x .^2, dims=2)
plot(1:N,sqrt.(xnorms), label = "||x||")


a = 0
b = 5
norms_range = range(a,b, length = 101)
histogram(xnorms/d, label="||x||", bins=norms_range, normalize=:pdf, color=:gray)
p(x) = 1/beta(d/2,nu/2)*(d/nu)^(d/2)*x^(d/2 - 1)*(1+d/nu*x)^(-(d+nu)/2)
plot!(norms_range, x -> p(x)/(beta_inc(d/2,nu/2,d*b/(d*b+nu))[1] -beta_inc(d/2,nu/2,d*a/(d*a+nu))[1]),
 label = "F", lw = 3)

beta_inc(d/2,nu/2,d*a/(d*a+nu))[2]

### Misc Tests

X = randn(10000,200)
plot(log.(1 .+ sqrt.(d*eigen(cov(X)).values)), label = "iid")
plot!(log.(1 .+ sqrt.(eigen(d*cov(out.x, corrected = false)).values)), label = "SBPS")
plot!(log.(1 .+ eigen(out.sigma[end]).values), label = "SBPS Est")
plot!(x -> log(1+sqrt(d)), label = "theoretical")



A = randn(3,3)
A = A'*A
eigentemp = eigen(A)

A2 = sqrt(A)

A - eigentemp.vectors*Diagonal(eigentemp.values)*eigentemp.vectors'

A2 - eigentemp.vectors*Diagonal(sqrt.(eigentemp.values))*eigentemp.vectors'
