### Main Testing File ###

    include("Adaptive SBPS.jl")
    include("Hamiltonian MC.jl")
    include("Adaptive SRW.jl")
    include("SHMC.jl")
    include("Adaptive Slice.jl")


    using ForwardDiff
    using Plots
    using SpecialFunctions
    using StatsBase
    using JLD

    d = 200
    nu = 2

    sigma = sqrt(d)I(d)
    mu = zeros(d) .+ 1e3

    d > 1 ? x0 = sigma*normalize(randn(d)) + mu : x0 = (sigma*rand([1,-1]))[1]

    #banana(x; b=0) = vcat(x[1] + b*x[2]^2,x[2:end])

    #f = x -> -sum(x.^2 ./(1 .+ abs.(x)))
    f = x -> -(nu+d)/2*log(nu + sum(x.^2))

    #b=0

    #f = x -> test(banana(x; b=b))
    #f = x -> test(banana(x; b=b))
    #f = test
    #f = x -> -sum(x.^2)/2

    #Set up gradient
    d > 1 ? gradlogf = x -> ForwardDiff.gradient(f,x) : gradlogf = x -> ForwardDiff.derivative(f,x)

    #This is here to precalculate the gradient function
    gradlogf(x0)
    
    #plot(eigen(cov(randn(100,d))).values)

    #sigma = (cov(randn(100,d)) + I(d))*sqrt(d)/2
    #mu = randn(d)

### SBPS Testing

    T = 10000 #1 hour 40 mins ish
    delta = 0.1
    Tbrent = pi/2
    Epsbrent = 0.01
    Abrent = 1.01
    Nbrent = 20
    tol = 1e-6
    lambda = 1 #5 gave best ACF for t_2, 0.6 gave best ACF for normal

    beta = 1.1
    burnin = T/5000
    adaptlength = T/5000
    R = 1e6
    r = 1e-3
    forgetrate = 3/4
    lambdageom = 0.1

    @time out = SBPSAdaptiveGeom(gradlogf, x0, lambda, T, delta, beta, r, R; Tbrent, Abrent, Nbrent, tol, sigma, mu, burnin, adaptlength, forgetrate, updategamma = true, updatelambda = false, lambdageom = 10);
    save("out.jld","out",out)

    out = load("out.jld")["out"]

    FullSBPSGeom = function(lambda)
        (zout,vout,eventsout,bounceratio,Nevals,Tout) = SBPSGeom(gradlogf, x0, lambda, T, delta; Tbrent = Tbrent, Abrent = Abrent, Nbrent = Nbrent, tol = tol,
        sigma = sigma, mu = mu);

        n = floor(BigInt, T/delta)+1 #Total number of observations of the skeleton path
        xout = zeros(n,d)

        #Project each entry back to R^d
        for i in 1:n
            xout[i,:] = SP(zout[i,:]; sigma = sigma, mu = mu)
        end

        return (z = zout, v = vout, x = xout, events = eventsout, bounceratio = bounceratio, Nevals = Nevals, Tbrent = Tout)
    end

    for lambda in 1:10
        @time out = FullSBPSGeom(lambda)

        save("out_$lambda.jld","out",out)
    end

    #@time out = FullSBPSGeom(lambda);

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

    plot(0:delta:T,out.z[:,end], label = "z_{d+1}")
    vline!(cumsum(out.times[1:end-1]), label = "Adaptations", lw = 0.5)

    plot(0:delta:T,sum(out.z[:,1:d], dims = 2)/sqrt(d), label = "z_{d+1}")
    vline!(cumsum(out.times[1:end-1]), label = "Adaptations")
    plot!(0:delta:T,out.z[:,1], label = "z_{d+1}")
    
    plot((0:1:700)*delta,autocor(out.x[:,1].^2, 0:1:700), label = "Autocorrelation of x_1^2")
    plot!(x -> 0, lwd = 3, label = "")

    plot((0:1:700)*delta,autocor(out.z[:,end], 0:1:700), label = "Autocorrelation of z_{d+1}")
    plot!(x -> 0, lwd = 3, label = "")

    plot((0:1:35000)*delta,autocor(out.x[:,1] .- b*out.x[2].^2, 0:1:35000), label = "Autocorrelation of x_1")
    plot!(x -> 0, lwd = 3, label = "")

    plot(cumsum(out.times),sum(out.Nevals, dims = 2) ./ out.times, label = "Gradient evaluations per unit time")

    #savefig("ASBPSevals.pdf")

    #map(x -> sum(x -> x^2, x - mu), eachrow(out.mu))
    #map(x -> sum(x -> x^2, eigen(x - sqrt(d)I(d)).values), out.sigma)

    myanim = @animate for i in 1:size(out.x)[1]
        myplot = plot(1, xlim = (-10,10), ylim = (-10,10), label="",framestyle=:origin)
        plot!(myplot, out.x[1:i,1], out.x[1:i,2], color=1, label="")
        scatter!(myplot, [out.x[i,1]], [out.x[i,2]], c=:red, label="")
        myplot
    end every 50

    gif(myanim)
    gif(myanim, "SBPS.mp4")

    #plot(out.x[:,1].^2,out.z[:,end])
    histogram2d(out.z[:,1], out.z[:,2], bins=(1000,1000),normalize=:pdf)

    mean(out.z[:,end])

    #savefig("ASBPSz.pdf")

    plot(sum(out.Nevals, dims=2)./out.times)


### SSS Tests

    Nslice::Int64 = 1000000
    stepsslice::Int64 = 3 #30 seconds

    beta = 1.1
    burninslice = Nslice/2000
    adaptlengthslice = Nslice/2000
    R = 1e9
    r = 1e-3
    forgetrate = 3/4
    
    @time sliceout = SliceAdaptive(f, x0, Nslice, beta, r, R; sigma, mu, burnin = burninslice, adaptlength = adaptlengthslice, steps = stepsslice, forgetrate = forgetrate);

    #@time sliceout = SliceSimulator(f, x0, Nslice; sigma, mu, steps = stepsslice);
    save("sliceout.jld","sliceout",sliceout)
    sliceout = load("sliceout.jld")["sliceout"]

    #Plot comparison against the true distribution
    p(x) = 1/sqrt(2pi)*exp(-x^2/2)
    #q(x) = 1/sqrt(2pi*sigmaf)*exp(-x^2/2sigmaf)
    q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
    b_range = range(-10,10, length=101)

    histogram(sliceout.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
    plot!(p, label= "N(0,1)", lw=3)
    plot!(q, label= "t", lw=3)
    xlabel!("x")
    ylabel!("P(x)")

    plot(1:stepsslice:Nslice*stepsslice,sliceout.x[:,1], label = "x1")
    vline!(stepsslice*cumsum(sliceout.times[1:end-1]), label = "Adaptations", lw = 0.5)

    plot(1:stepsslice:Nslice*stepsslice,sliceout.z[:,end], label = "z_{d+1}")
    vline!(stepsslice*cumsum(sliceout.times[1:end-1]), label = "Adaptations", lw = 0.5)

    plot(stepsslice*cumsum(sliceout.times), sliceout.Nprop ./ (stepsslice*sliceout.times), label = "Proposals per step")

    savefig("ASSSprop.pdf")

    plot((0:1:700)*stepsslice,autocor(sliceout.x[:,1].^2, (0:1:700)), label="Autocorrelation of x_1^2")
    plot!(x -> 0, lwd = 3, label="")

    plot((0:1:700)*stepsslice,autocor(sliceout.z[:,end], (0:1:700)), label="Autocorrelation of z_{d+1}")
    plot!(x -> 0, lwd = 3, label="")

    plot((0:1:35000)*stepsslice,autocor(sliceout.x[:,1] .- b*sliceout.x[2].^2, 0:1:35000), label = "Autocorrelation of x_1")
    plot!(x -> 0, lwd = 3, label = "")

    #savefig("SSSautocorbanana2.pdf")
    #map(x -> sum(x -> x^2, x), eachrow(sliceout.mu))
    #plot(log.(map(x -> sum(x -> x^2, eigen(x - sqrt(d)I(d)).values), sliceout.sigma)))

    #plot(sliceout.x[:,1],sliceout.x[:,2])
    histogram2d(sliceout.x[:,1], sliceout.x[:,2], bins=(1000,1000),normalize=:pdf)


    myanim = @animate for i in 1:size(sliceout.x)[1]
        myplot = plot(1, xlim = (-10,10), ylim = (-10,10), label="",framestyle=:origin)
        plot!(myplot, sliceout.x[1:i,1], sliceout.x[1:i,2], color=1, label="")
        scatter!(myplot, [sliceout.x[i,1]], [sliceout.x[i,2]], c=:red, label="")
        myplot
    end every 5

    gif(myanim)
    gif(myanim, "SSS.mp4")


### SRW Tests

    h = d^-1
    Nsrw::Int64 = 1000000
    stepssrw::Int64 = 1500 #6 hours 20mins ish
    
    beta = 1.1
    burninsrw = Nsrw/8000
    adaptlengthsrw = Nsrw/8000
    R = 1e9
    r = 1e-3
    forgetrate = 3/4
    hgeom = 10

    @time srwout = SRWAdaptive(f, x0, h, Nsrw, beta, r, R; sigma, mu,burnin = burninsrw, adaptlength = adaptlengthsrw, steps = stepssrw, forgetrate = forgetrate, updategamma = true, updateh = true, hgeom = hgeom);

    #@time srwout = SRWSimulator(f, x0, h, Nsrw; sigma, mu, steps = stepssrw);
    srwout.a

    save("srwout.jld","srwout",srwout)
    #srwout = load("srwout.jld")["srwout"]

    #p(x) = 1/sqrt(2pi)*exp(-x^2/2)
    q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
    b_range = range(-8,8, length=101)

    histogram(srwout.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
    plot!([q p], label= ["t" "N(0,1)"], lw=3)
    #plot!(q, label= "t", lw=3)
    xlabel!("x")
    ylabel!("P(x)")

    plot((1:Nsrw)*stepssrw, srwout.x[:,1], label = "x1")
    vline!(stepssrw*cumsum(srwout.times[1:end-1]), label = "Adaptations", lw = 0.5)

    plot((1:Nsrw)*stepssrw,srwout.z[:,end], label = "z_{d+1}", legend=:top)
    vline!(stepssrw*cumsum(srwout.times[1:end-1]), label = "Adaptations", lw = 0.5)


    plot((0:1:700)*stepssrw,autocor(srwout.x[:,1].^2, 0:1:700), label="Autocorrelation of x_1")
    plot!(x -> 0, lwd = 3, label="")

    plot((0:1:700)*stepssrw,autocor(srwout.z[:,end], 0:1:700), label="Autocorrelation of z_{d+1}")
    plot!(x -> 0, lwd = 3, label="")

    #savefig("ASRWz.pdf")

    #plot(srwout.x[:,1],srwout.x[:,2])
    histogram2d(srwout.x[:,1], srwout.x[:,2], bins=(1000,1000),normalize=:pdf)

    srwxnorms = vec(sum(srwout.x .^2, dims=2))
    #plot(sqrt.(srwxnorms), label = "||x||")
    maximum(srwxnorms)

    myanim = @animate for i in 1:size(srwout.x)[1]
        myplot = plot(1, xlim = (-10,10), ylim = (-10,10), label="",framestyle=:origin)
        plot!(myplot, srwout.x[1:i,1], srwout.x[1:i,2], color=1, label="")
        scatter!(myplot, [srwout.x[i,1]], [srwout.x[i,2]], c=:red, label="")
        myplot
    end every 5

    gif(myanim)
    gif(myanim, "SRW.mp4")

### HMC Testing

    hmcdelta = 2*d^(-1/4)
    L = 5
    d > 1 ? M = I(d) : M = 1
    Nhmc::Int64 = 1000000
    hmcsteps = 9 #5 gave 2000sec for b=0, 9 gave 3000sec for b=1

    hmcout = zeros(Nhmc,d)

    stepstest = zeros(1)
    stepstest[1] = hmcsteps
    timedif = 3000

    for i in 1:5
        start_time = Int(time_ns())
        @time hmcout = HMC(f, gradlogf, x0, Nhmc, hmcdelta, L; M = M, steps = hmcsteps);
        #hmcout.a
        end_time = Int(time_ns())

        timedif = (end_time-start_time)/1e9
        println(timedif)
        if (timedif < 2800) || (timedif > 3200)
            hmcsteps = ceil(Int64, hmcsteps * 3000/timedif)
            append!(stepstest,hmcsteps)
        else
            break
        end
    end

    save("hmcout.jld","hmcout",hmcout)
    #hmcout = load("hmcbanana.jld")["hmcout"]

        
    #Plot comparison against the true distribution
    p(x) = 1/sqrt(2pi)*exp(-x^2/2)
    q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
    b_range = range(-10,10, length=101)

    histogram(hmcout.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
    plot!([p q], label= ["N(0,1)" "t"], lw=3)
    #plot!(q, label= "t", lw = 3)
    xlabel!("x")
    ylabel!("P(x)")

    plot(1:hmcsteps:hmcsteps*Nhmc,hmcout.x[:,1], label = "x1")

    hmcoutz = zeros(Nhmc,d+1)
    for i in 1:Nhmc
        hmcoutz[i,:] = SPinv(hmcout.x[i,:]; sigma = sigma, mu = mu)
    end

    plot((0:1:5000)*hmcsteps,autocor(abs.(hmcout.x[:,1]), 0:1:5000), label="Autocorrelation of x_1")
    plot!(x -> 0, lwd = 3, label="")

    plot(0:1:20000,autocor(hmcoutz[:,end], 0:1:20000), label="Autocorrelation of z_{d+1}")
    plot!(x -> 0, lwd = 3, label="")


    #savefig("HMCautocorbanana2.pdf")


    #plot(hmcout.x[:,1],hmcout.x[:,2])
    histogram2d(hmcout.x[:,1], hmcout.x[:,2], bins=(1000,1000),normalize=:pdf)

    hmcxnorms = vec(sum(hmcout.x .^2, dims=2))
    plot(1:hmcsteps:hmcsteps*Nhmc,sqrt.(hmcxnorms), label = "||x||")
    #maximum(hmcxnorms)




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
        @time srwout = SRWAdaptive(f, x0, h, Nsrw, beta, r, R; sigma, mu, burnin = burninsrw, adaptlength = adaptlengthsrw);

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
    z = range(-1,1,length=2001)[2:end-1]
    radius(d) = d^1.3

    for d in [1,5,10,50,100,1000]
        stephist!(p,collect(z), weights= map(z -> exp(-radius(d)/(1-z) -d*log(1-z) +d/2*log(1-z^2) + (d+radius(d))/2 -d/2*log(d) +d/2*log(radius(d))),z),
        bins=z,normalize=:pdf,label="d=$d",lwd=3)
    end
    plot!(xlim=xlims(), ylim=ylims())

    for d in [10000]
        stephist!(p,collect(z), weights= map(z -> exp(-radius(d)/(1-z) -d*log(1-z) +d/2*log(1-z^2) + (d+radius(d))/2 -d/2*log(d) +d/2*log(radius(d))),z),
        bins=z,normalize=:pdf,label="d=$d",lwd=3)
    end

    p 

    savefig("NormLatitudeDensitiesBigVar.pdf")