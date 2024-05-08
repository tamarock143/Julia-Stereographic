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
mu = zeros(d)

nu = 2

d > 1 ? x0 = sigma*normalize(randn(d)) + mu : x0 = (sigma*rand([1,-1]) + mu)[1]

f = x -> -(nu + d)/2*log(nu+sum(x.^2))

d > 1 ? gradlogf = x -> ForwardDiff.gradient(f,x) : gradlogf = x -> ForwardDiff.derivative(f,x)

#This is here to precalculate the gradient function
gradlogf(x0)


### SBPS Testing

T = 2.2e6 #~1000sec
delta = 0.1
Tbrent = pi/10
Epsbrent = 0.01
tol = 1e-6
lambda = 1

beta = 1.1
burnin = T/200
adaptlength = T/200
R = 1e9
r = 1e-3

@time out = SBPSAdaptive(gradlogf, x0, lambda, T, delta, beta, r, R; Tbrent, Epsbrent, tol, sigma, mu, burnin, adaptlength);

FullSBPS = function()
    (zout,vout,eventsout) = SBPSSimulator(gradlogf, x0, lambda, T, delta; Tbrent = Tbrent, Epsbrent = Epsbrent, tol = tol,
    sigma = sigma, mu = mu);

    n = floor(BigInt, T/delta)+1 #Total number of observations of the skeleton path
    xout = zeros(n,d)

    #Project each entry back to R^d
    for i in 1:n
        xout[i,:] = SP(zout[i,:]; sigma = sigma, mu = mu)
    end

    return (z = zout, v = vout, x = xout, events = eventsout)
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

plot(0:delta:T,out.x[:,1], label = "x1")
vline!(cumsum(out.times[1:end-1]), label = "Adaptations", lw = 0.5)
plot!(0:delta:T,out.x[:,2], label = "x2")

#map(x -> sum(x -> x^2, x - mu), eachrow(out.mu))
#map(x -> sum(x -> x^2, eigen(x - sqrt(d)I(d)).values), out.sigma)

eventtimes = findall(x -> x>1e-2, vec(sum(x -> x^2,out.v[1:end-1,:] - out.v[2:end,:], dims=2)))


myplot = plot(1, xlim = (-3.5,3.5), ylim = (-3.5,3.5), label="SBPS path",legend=:topleft,framestyle=:origin)
myanim = @animate for i in 1:size(out.x)[1]
    push!(myplot, out.x[i,1],out.x[i,2])
end every 10

gif(myanim, "SBPS.gif")

#plot(out.x[:,1],out.x[:,2])

plot(0:delta:T,out.z[:,end], label = "z_{d+1}")
vline!(cumsum(out.times[1:end-1]), label = "Adaptations")

mean(out.z[:,end])

#savefig("tAdaptationsLatitude.pdf")


### SSS Tests

Nslice::Int64 = 2.5e7 #~1000sec

beta = 1.1
burninslice = Nslice/200
adaptlengthslice = Nslice/200
R = 1e9
r = 1e-3

@time sliceout = SliceAdaptive(f, x0, Nslice, beta, r, R; sigma, mu, burnin = burninslice, adaptlength = adaptlengthslice);

#@time sliceout = SliceSimulator(f, x0, Nslice; sigma, mu);

#Plot comparison against the true distribution
#p(x) = 1/sqrt(2pi)*exp(-x^2/2)
#q(x) = 1/sqrt(2pi*sigmaf)*exp(-x^2/2sigmaf)
q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
b_range = range(-8,8, length=101)

histogram(sliceout.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
#plot!(p, label= "N(0,1)", lw=3)
plot!(q, label= "t", lw=3)
xlabel!("x")
ylabel!("P(x)")

plot(sliceout.x[:,1], label = "x1")
vline!(cumsum(sliceout.times[1:end-1]), label = "Adaptations", lw = 0.5)

plot(sliceout.z[:,end], label = "z_{d+1}")
vline!(cumsum(sliceout.times[1:end-1]), label = "Adaptations", lw = 0.5)

#map(x -> sum(x -> x^2, x), eachrow(sliceout.mu))
#plot(log.(map(x -> sum(x -> x^2, eigen(x - sqrt(d)I(d)).values), sliceout.sigma)))

plot(autocor(sliceout.x[:,1].^2))

cov(sliceout.x[floor(Int64,2/3*10^5):end,:])




### SRW Tests

h2 = 20*d^-1
Nsrw::Int64 = 2.4e7 #~1000 seconds

beta = 1.1
burninsrw = Nsrw/200
adaptlengthsrw = Nsrw/200
R = 1e9
r = 1e-3

@time srwout = SRWAdaptive(f, x0, h2, Nsrw, beta, r, R; sigma, mu,burnin = burninsrw, adaptlength = adaptlengthsrw);

#@time srwout = SRWSimulator(f, x0, h2, Nsrw; sigma, mu);

#p(x) = 1/sqrt(2pi)*exp(-x^2/2)
q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
b_range = range(-8,8, length=101)

histogram(srwout.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
#plot!([q p], label= ["t" "N(0,1)"], lw=3)
plot!(q, label= "t", lw=3)
xlabel!("x")
ylabel!("P(x)")

plot(srwout.x[:,1])
vline!(cumsum(out.times[1:end-1]), label = "Adaptations", lw = 0.5)

#plot(srwout.x[:,1],srwout.x[:,2])

srwxnorms = vec(sum(srwout.x .^2, dims=2))
#plot(sqrt.(srwxnorms), label = "||x||")
maximum(srwxnorms)



### HMC Testing

hmcdelta = 1.45d^(-1/4)
L = 5
d > 1 ? M = I(d) : M = 1
N::Int64 = 1.8e7 #~1000sec

@time hmcout = HMC(f, gradlogf, x0, N, hmcdelta, L; M = M);
hmcout.a
#Plot comparison against the true distribution
#p(x) = 1/sqrt(2pi)*exp(-x^2/2)
q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
b_range = range(-8,8, length=101)

histogram(hmcout.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
#plot!([q p], label= ["t" "N(0,1)"], lw=3)
plot!(q, label= "t", lw = 3)
xlabel!("x")
ylabel!("P(x)")

plot(hmcout.x[:,1], label = "x")

#plot(hmcout.x[:,1],hmcout.x[:,2])

hmcxnorms = vec(sum(hmcout.x .^2, dims=2))
#plot(sqrt.(hmcxnorms), label = "||x||")
maximum(hmcxnorms)




### Misc Tests

#P(|T| > a) for T ~ t_nu in d dimensions
Z(a) = beta_inc(d/2,nu/2,d*a/(d*a+nu))[2]

#Output norms of SBPS process
mySBPStest = function ()
    @time out = SBPSAdaptive(gradlogf, x0, lambda, T, delta, beta, r, R; Tbrent, Epsbrent, tol, sigma, mu, burnin, adaptlength);

    xnorms = vec(sum(out.x.^2, dims=2))

    return(xnorms)
end

#Output norms of SRW process
mySRWtest = function ()
    @time srwout = SRWAdaptive(f, x0, h2, Nsrw, beta, r, R; sigma, mu, burnin = burninsrw, adaptlength = adaptlengthsrw);

    srwxnorms = vec(sum(srwout.x.^2, dims=2))
    
    return(srwxnorms)
end

#Output norms of HMC process
myHMCtest = function ()
    @time hmcout = HMC(f, gradlogf, x0, N, hmcdelta, L; M = M)

    hmcxnorms = vec(sum(hmcout.x .^2, dims=2))
    
    return(hmcxnorms)
end

#Output norms of SSS process
mySSStest = function ()
    @time sliceout = SliceAdaptive(f, x0, Nslice, beta, r, R; sigma, mu, burnin = burninslice, adaptlength = adaptlengthslice);

    slicexnorms = vec(sum(sliceout.x .^2, dims=2))
    
    return(slicexnorms)
end

p = plot(10 .^(-2:0.1:9), a -> abs(sum(xnorms/d .>= a)/length(xnorms) - Z(a))/Z(a), label = "SBPS")
plot!(p, xscale=:log10, yscale=:log10, minorgrid=true)
plot!(p, 10 .^(-2:0.1:9), a -> abs(sum(srwxnorms/d .>= a)/length(srwxnorms) - Z(a))/Z(a), label = "SRW")
plot!(p, 10 .^(-2:0.1:9), a -> abs(sum(hmcxnorms/d .>= a)/length(hmcxnorms) - Z(a))/Z(a), label = "HMC")
plot!(p, 10 .^(-2:0.1:9), a -> abs(sum(slicexnorms/d .>= a)/length(slicexnorms) - Z(a))/Z(a), label = "Slice")
plot!(p, legend=:bottomright)
title!(p, "Log Absolute Relative Error for CCDF of norm of a t-distribution\nwith d = 2,  ν = 2 (Runtime of ~1000 seconds)", titlefontsize = 10)

savefig("tNormDistComparisonSSS.pdf")


out = load("sbps.jld")["SBPS"]
hmcout = load("hmc.jld")["HMC"]

xnorms = load("xnorms.jld")["xnorms"]
hmcxnorms = load("hmcxnorms.jld")["hmcxnorms"]

srwxnorms = load("srwxnorms.jld")["srwxnorms"]
slicexnorms = load("slicexnorms.jld")["slicexnorms"]

p = plot()
z = range(-1,1,length=1001)[2:end-1]

for d in [1,2,5,10,50,100,200]
    stephist!(p,collect(z), weights= map(z -> exp(-d/(1-z) -d*log(1-z) +d/2*log(1-z^2)),z),
     bins=x,normalize=:pdf,label="d=$d",lwd=3)
end
p

savefig("NormLatitudeDensitiesSmallVar.pdf")