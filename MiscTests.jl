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