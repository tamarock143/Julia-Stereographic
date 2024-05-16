#Setup for Big Test
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

nu = 1.6

d > 1 ? x0 = sigma*normalize(randn(d)) + mu : x0 = (sigma*rand([1,-1]) + mu)[1]

f = x -> -(nu + d)/2*log(nu+sum(x.^2))

d > 1 ? gradlogf = x -> ForwardDiff.gradient(f,x) : gradlogf = x -> ForwardDiff.derivative(f,x)

#This is here to precalculate the gradient function
gradlogf(x0)


beta = 1.1
R = 1e9
r = 1e-3


### SBPS Testing

T = 2.2e6 #~1000sec
delta = 0.1
Tbrent = pi/10
Epsbrent = 0.01
tol = 1e-6
lambda = 1

burnin = T/200
adaptlength = T/200


### SSS Testing

Nslice::Int64 = 2.5e7 #~1000sec

burninslice = Nslice/200
adaptlengthslice = Nslice/200


### SRW Tests

h2 = 20*d^-1
Nsrw::Int64 = 2.45e7 #~1000 seconds

burninsrw = Nsrw/200
adaptlengthsrw = Nsrw/200


### HMC Testing

hmcdelta = 1.45d^(-1/4)
L = 5
d > 1 ? M = I(d) : M = 1
N::Int64 = 1.85e7 #~1000sec




#P(|T| > a) for T ~ t_nu in d dimensions
Z(a) = beta_inc(d/2,nu/2,d*a/(d*a+nu))[2]

#Output norms of SBPS process
mySBPStest = function ()
    @time out = SBPSAdaptive(gradlogf, x0, lambda, T, delta, beta, r, R; Tbrent, Epsbrent, tol, sigma, mu, burnin, adaptlength);

    xnorms = vec(sum(out.x.^2, dims=2))

    return(xnorms)
end

#Output norms of SSS process
mySSStest = function ()
    @time sliceout = SliceAdaptive(f, x0, Nslice, beta, r, R; sigma, mu, burnin = burninslice, adaptlength = adaptlengthslice);

    slicexnorms = vec(sum(sliceout.x .^2, dims=2))
    
    return(slicexnorms)
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


#### TESTING RANGE
Ntest = 10

xnorms = reduce(hcat,[mySBPStest() for _ in 1:Ntest])
srwxnorms = reduce(hcat,[mySRWtest() for _ in 1:Ntest])
slicexnorms = reduce(hcat,[mySSStest() for _ in 1:Ntest])
hmcxnorms = reduce(hcat,[myHMCtest() for _ in 1:Ntest])


p = plot(10 .^(-2:0.1:11), a -> abs(sum(xnorms/d .>= a)/length(xnorms) - Z(a))/Z(a), label = "SBPS")
plot!(p, xscale=:log10, yscale=:log10, minorgrid=true)
plot!(p, 10 .^(-2:0.1:11), a -> abs(sum(srwxnorms/d .>= a)/length(srwxnorms) - Z(a))/Z(a), label = "SRW")
plot!(p, 10 .^(-2:0.1:11), a -> abs(sum(hmcxnorms/d .>= a)/length(hmcxnorms) - Z(a))/Z(a), label = "HMC")
plot!(p, 10 .^(-2:0.1:11), a -> abs(sum(slicexnorms/d .>= a)/length(slicexnorms) - Z(a))/Z(a), label = "Slice")
plot!(p, legend=:bottomright)
title!(p, "Log Absolute Relative Error for CCDF of norm of a t-distribution\nwith d = 2,  ν = 1.6 (10 runs each of ~1000 seconds)", titlefontsize = 10)
p

savefig("tNormDistComparisonNu16.pdf")