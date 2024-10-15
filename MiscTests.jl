bouncehess = SBPSRateDeriv(hessianlogf,gradlogf)

p = plot(0:0.001:Tbrent, s -> bouncerate(s,z,v; sigma = sigma, mu = mu)[1], lw = 3)

for t in 0.05
    (c,m) = bouncehess(t,z,v; sigma = sigma, mu = mu)[(:rate, :derivative)]
    
    println([c,m])
    plot!(p,0:0.001:Tbrent,x -> m*(x - t) + c)
end

plot(p)

Tangentmin()