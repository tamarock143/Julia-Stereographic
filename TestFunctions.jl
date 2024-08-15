### Cauchy Regression Model

d = 11
a = b = 0.1

(u0,alpha0,beta0...) = vcat(0,-2,-4:4)

ndata = 15
mydata = randn(ndata,d-2)

myobs = exp(u0)*tan.(pi*rand(ndata) .- 1/2) .+ alpha0 .+ mydata*beta0

f(p) = (a - ndata)*p[1] - b*exp(p[1]) - sum(log,1 .+ exp(-2p[1])*(myobs .- p[2] .- mydata*p[3:end]).^2)

gradlogf = x -> ForwardDiff.gradient(f,x)

gradlogf(vcat(u0,alpha0,beta0))