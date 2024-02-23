include("Adaptive SBPS.jl")
include("Hamiltonian MC.jl")

using ForwardDiff
using Plots
using SpecialFunctions
using StatsBase

d = 200
sigma = 1e4*sqrt(d)I(d)
mu = zeros(d)
nu = 200

f = x -> log(1+ sum(x.^2)/nu)*-((nu+d)/2)

d > 1 ? x0 = sigma*normalize(randn(d)) + mu : x0 = sigma*rand([1,-1]) + mu

d > 1 ? gradlogf = x -> ForwardDiff.gradient(f,x) : gradlogf = x -> ForwardDiff.derivative(f,x)

#This is here to precalculate the gradient function
gradlogf(x0)


### SBPS Testing

T = 1e4
delta = 0.2
Tbrent = pi/200
Epsbrent = 0.01
tol = 1e-6
lambda = 1

beta = 1.1
burnin = 50
adaptlength = 50
R = 1e9
r = 1e-6

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

plot(0:delta:T,out.x[:,1], label = "x")
vline!(cumsum(out.times[1:end-1]), label = "Adaptations", lw = 0.5)

map(x -> sum(x -> x^2, x - mu), eachrow(out.mu))
map(x -> sum(x -> x^2, eigen(x - sqrt(d)I(d)).values), out.sigma)

plot(out.x[:,1],out.x[:,2])

plot(autocor(out.x[:,1]))

xnorms = sum(out.x.^2, dims=2)
plot(sqrt.(xnorms), label = "||x||")
vline!(cumsum(out.times[1:end-1]), label = "Adaptations")

latf(x,theta) = -(x - theta)/(x + theta) #Latitude of a given z at position ||x||^2/theta
plot(0:0.1:10, theta -> mean(x -> latf(x,theta), xnorms))
hline!([0])
c = RobMonro(latf, xnorms, d, 1, 1e6; lower = 1, upper = d^2)

latfm(theta) = mean(x -> latf(x,theta), xnorms)
latfg(theta) = -mean(x -> 2x/(x+theta)^2, xnorms)
cn = Newton(latfm,latfg, d, tol)
vline!([cn])

latf2(theta) = sum(x -> ((x - theta)/(x + theta))^2, xnorms)/length(xnorms) #Mean of Squared Latitude of a given z at position ||x||^2/theta
c2 = Brent(latf2, 1, d^2, tol)

plot(0:0.1:10,theta -> sum(latf.(xnorms,theta))/length(xnorms))

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

mean(out.z[:,end])

savefig("tAdaptationsLatitude.pdf")


### HMC Testing

delta = 1.45d^(-1/4)
L = 5
d > 1 ? M = I(d) : M = 1
N::Int64 = 1e6

@time hmcout = HMC(f, gradlogf, x0, N, delta, L; M = M);
hmcout.a
#Plot comparison against the true distribution
p(x) = 1/sqrt(2pi)*exp(-x^2/2)
q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
b_range = range(-8,8, length=101)

histogram(hmcout.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
plot!([q p], label= ["t" "N(0,1)"], lw=3)
xlabel!("x")
ylabel!("P(x)")

plot(hmcout.x[collect(Int64,1:1e0:N),1], label = "x")

plot(autocor(hmcout.x[:,1]))

plot(hmcout.x[:,1],hmcout.x[:,2])

hmcxnorms = sum(hmcout.x .^2, dims=2)
plot(sqrt.(hmcxnorms[1:1000000]), label = "||x||")

### Misc Tests

plot(zpath[:,end])
plot!(map(x -> (x-d)/(x+d),xnorms))