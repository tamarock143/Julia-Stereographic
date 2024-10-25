bouncehess = SBPSRateDeriv(hessianlogf,gradlogf)

p = plot(0:0.001:Tbrent, s -> bouncerate(s,z,v; sigma = sigma, mu = mu)[1], lw = 3)

for t in 0.7
    (c,l) = bouncerate(t,z,v; sigma = sigma, mu = mu)[(:rate,:grad)]

    temphess = testhess(z*cos(t)+v*sin(t))

    m = -sum(z .* l) + sum(v'*temphess*v)
    
    println([c,m])
    plot!(p,0:0.001:Tbrent,x -> m*(x - t) + c)
end

plot(p)

newpi(z) = f(SP(normalize(z); sigma, mu)) - d*log(1-normalize(z)[end]) - d/2*log(sum(z.^2))

testhess = z -> ForwardDiff.hessian(newpi,z)

testhess(z)

testderiv = s -> ForwardDiff.derivative(t -> bouncerate(t,z,v; sigma = sigma, mu = mu)[:rate], s)









N = 1000000
myx = randn(N)
theta = 0.5

myy = zeros(N+1)
for i in 1:N
    myy[i+1] = theta*myy[i] + myx[i]
end

myz = zeros(1000+1)
for i in 1:1000
    myz[i+1] = theta*myz[i] + myx[i]
end

plot(autocor(myy[1:20000], 0:30))
plot!(x -> 2*theta^x)
plot(autocor(myz[1:1001], 0:30))
