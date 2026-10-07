### Tests for Barker MCMC 

#Import libraries for testing
    include("Adaptive SBPS.jl")
    include("Hamiltonian MC.jl")
    include("Adaptive SRW.jl")
    include("SHMC.jl")
    include("Adaptive Slice.jl")
    include("SBarker.jl")
    include("Adaptive Barker.jl")

    using ForwardDiff
    using Plots
    using SpecialFunctions
    using StatsBase
    using JLD
    using LaTeXStrings
    using Distributions
    
#Optimal Scaling Tests
    d = 50000
    l = 1
    h = l*d^(-1/6)

    logf = x -> -sum(x.^2)/2
    gradlogf = x -> -x

    #Testing various things
    barkertest = function(reps)
        out = zeros(13)
        
        for _ in 1:reps
            x = randn(d) #Position vector, initialised at x0
            
            fx = logf(x) #Precalculate density at position
            gradx = gradlogf(x) #Precalculate gradient at position
            normgradx = norm(gradx) #Precalculate norm(gradient) at position
            

            v = h*randn(d)

            #Flip steps
            y = BarkerStep(normgradx/sqrt(d)*ones(d),v)

            delta = Rotategrad1inv(gradx, y)

            #Proposal position, log-density and gradient
            xprime = x + delta

            fxprime = logf(xprime)
            gradxprime = gradlogf(xprime)

            #Calculate reverse step
            yprime = Rotategrad1(gradxprime,x-xprime)

            #Compute acceptance probability
            a = fxprime - fx + sum(log.(1 .+ exp.(-y*norm(gradx)/sqrt(d)))) - sum(log.(1 .+ exp.(-yprime*norm(gradxprime)/sqrt(d))))

            denom = 1
            num1 = sum(y)/sqrt(d)

            num2 = sum(y)/sqrt(d)

            deltatemp = y - num1/denom*x/sqrt(d) - num2/denom*ones(d)/sqrt(d)

            testemp = y - l^2*d^(-1/3)*(x+ones(d)+delta/2)
            testemp2 = (1-l^2*d^(-1/3)/2)*y - (x+ones(d))*l^2*d^(-1/3)*(1-l^2*d^(-1/3)/4)

            tempfinal = (1-l^2*d^(-1/3)/2)*y - (x+ones(d))*l^2*d^(-1/3)#*(1-l^2*d^(-1/3)/4)

            out[1] += a/reps
            out[2] += a^2/reps

            out[3] += (-l^4 * d^(1/3)/8 + l^2 *d^(-1/3)/4 * sum(y) - sum(y.^4 - yprime.^4)/192)/reps
            out[4] += (-l^4 * d^(1/3)/8 + l^2 *d^(-1/3)/4 * sum(y) - sum(y.^4 - yprime.^4)/192)^2/reps

            out[5] += (-l^4 * d^(1/3)/8 + l^2 *d^(-1/3)/4 * sum(y))/reps
            out[6] += (-l^4 * d^(1/3)/8 + l^2 *d^(-1/3)/4 * sum(y))^2/reps

            ysum = sum(x -> x^4, y)

            out[7] += (ysum - sum(yprime.^4))/reps/192
            out[8] += (ysum - sum(tempfinal.^4))/reps/192
            out[9] += (ysum - sum(testemp.^4))/reps/192
            out[10] += (ysum - sum(testemp2.^4))/reps/192
            out[11] += (ysum - sum(yprime.^4))^2/reps/192^2

            #out[12] += sum(delta - deltatemp)/reps
            #out[13] += (sum(delta - deltatemp))^2/reps
        end

        out[2] -= out[1]^2
        out[4] -= out[3]^2
        out[6] -= out[5]^2
        out[11] -= out[7]^2

        println("Theoretical Mean and Variance")
        println([[-l^6/32] [l^6/16]])
        println("Log-Acceptance Mean and Variance")
        println(out[1:2]')

        println()

        println("Approximation Mean and Variance")
        println(out[3:4]')

        println()

        println("Just second order")
        println(out[5:6]')

        println()

        println("4th order comparison: (y')^4, tempfinal, testemp, testemp2")
        println(out[7:10]')

        println()
        println("4th order variance:")
        println(out[11])

        #out[12] = out[7]
        #out[13] = out[7]

        #println()
        #println("Diagnosing difference")
        #println(out[12:13])

        return(out)
    end

    mytest = barkertest(10000);

#Optimal Scaling Plots

    N = 10000
    d = 250
    l = 1
    h = l*d^(-1/6)

    x0 = randn(d)

    reps = 5000
    hspan = 10 .^(-2:0.1:1)
    nu = 100

    mu = zeros(d)
    sigma = sqrt(d)*I(d)

    #logf = x -> -sum(x.^2)/2
    #logf = x -> -sum(y -> sqrt(y^2 + 1e-6), x)
    #logf = x -> -(nu+d)/2*log(nu + sum(x.^2))
    #logf = x -> -sum(x.^4)
    logf = x -> -(nu+1)/2*sum(y -> log(nu + y^2), x)

    #Set up gradient
    d > 1 ? gradlogf = x -> ForwardDiff.gradient(logf,x) : gradlogf = x -> ForwardDiff.derivative(logf,x)
    #gradlogf = x -> -x
    #gradlogf = x -> -(nu+d)*x/(nu + sum(x.^2))

    #This is here to precalculate the gradient function
    gradlogf(x0)

    #out = SMALA(logf, x0, h, N; gradlogf = gradlogf, sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)), includefirst = true, steps = 1, printing = false)
    #out2 = StereoBarkerSim(logf, x0, h, N; gradlogf = gradlogf, sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)), includefirst = true, steps = 1, printing = false)


    lighttailsim = function(n)
        out = zeros(n)
        for i in 1:n
            flag = false
            while flag == false
                x = randn(Float64)
                u = rand()

                if exp(x^2/2 - x^4) > u/0.9
                    out[i] = x
                    flag = true
                end
            end
        end

        return out
    end
    lightvar = sum(lighttailsim(N).^2)/(N-1)


    #histograms
        p(x) = 1/sqrt(2pi)*exp(-x^2/2)
        q(x) = 1/(2gamma(7/6))*exp(-x^6)
        #q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
        b_range = range(-10,10, length=101)

        histogram(out.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
        #histogram(lighttailsim(100000), label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
        plot!(p, label= "N(0,1)", lw=3)
        plot!(q, label= "light", lw=3)
        xlabel!("x")
        ylabel!("P(x)")
    

    #Testing various things
    rotatetest = function(reps,h)

        aout = 0
        esjd = 0
        
        for _ in 1:reps
            x = rand(TDist(nu), d) #Position vector, initialised at x0

            fx = logf(x) #Precalculate density at position
            gradx = gradlogf(x) #Precalculate gradient at position
            normgradx = norm(gradx) #Precalculate norm(gradient) at position

            v = h*randn(d)

            #Flip steps
            y = BarkerStep(normgradx/sqrt(d)*ones(d),v)

            delta = Rotategrad1inv(gradx, y)

            #Proposal position, log-density and gradient
            xprime = x + delta

            fxprime = logf(xprime)
            gradxprime = gradlogf(xprime)

            #Calculate reverse step
            yprime = Rotategrad1(gradxprime,x-xprime)

            #Compute acceptance probability
            a = fxprime - fx + sum(log1_exp.(-y*norm(gradx)/sqrt(d))) - sum(log1_exp.(-yprime*norm(gradxprime)/sqrt(d)))
        
            aout += min(1,exp(a))

            esjd += min(1,exp(a))*norm(delta)^2
        end

        return([aout/reps, esjd/reps])
    end

    bigrotatetest = function(reps, hspan)
        n = length(hspan)

        out = zeros(n,2)

        for i in 1:n
            out[i,:] .= rotatetest(reps,hspan[i])
        end

        return out
    end

    rotateout = bigrotatetest(reps, hspan)

    myp = plot()
    plot!(myp, rotateout[:,1], rotateout[:,2], label="Rotate Barker")


    malatest = function(reps,h)

        aout = 0
        esjd = 0
        
        for _ in 1:reps
            x = rand(TDist(nu), d) #Position vector, initialised at x0

            fx = logf(x) #Precalculate density at position
            gradx = gradlogf(x) #Precalculate gradient at Position

            v = randn(d)

            #Proposal position, log-density and gradient
            xprime = x + h^2/2*gradx + h*v

            fxprime = logf(xprime)
            gradxprime = gradlogf(xprime)

            #Compute acceptance probability
            a = fxprime - fx - 1/(2h^2)*norm(x - xprime - h^2/2*gradxprime)^2 + 1/(2h^2)*norm(xprime - x - h^2/2*gradx)^2
            
            aout += min(1,exp(a))

            esjd += min(1,exp(a))*norm(x - xprime)^2
        end

        return([aout/reps, esjd/reps])
    end
    
    bigmalatest = function(reps, hspan)
        n = length(hspan)

        out = zeros(n,2)

        for i in 1:n
            out[i,:] .= malatest(reps,hspan[i])
        end

        return out
    end
    
    malaout = bigmalatest(reps, hspan)

    plot!(myp, malaout[:,1], malaout[:,2], label="MALA")
    

    coordtest = function(reps,h)

        aout = 0
        esjd = 0
        
        for _ in 1:reps
            x = rand(TDist(nu), d) #Position vector, initialised at x0

            fx = logf(x) #Precalculate density at position
            gradx = gradlogf(x) #Precalculate gradient at position

            v = h*randn(d)

            #Flip steps
            y = BarkerStep(gradx,v)

            #Proposal position, log-density and gradient
            xprime = x + y

            fxprime = logf(xprime)
            gradxprime = gradlogf(xprime)

            #Compute acceptance probability
            a = fxprime - fx + sum(log1_exp.(-y .* gradx)) - sum(log1_exp.(y .* gradxprime))
        
            aout += min(1,exp(a))

            esjd += min(1,exp(a))*norm(y)^2
        end

        return([aout/reps, esjd/reps])
    end

    bigcoordtest = function(reps, hspan)
        n = length(hspan)

        out = zeros(n,2)

        for i in 1:n
            out[i,:] .= coordtest(reps,hspan[i])
        end

        return out
    end

    coordout = bigcoordtest(reps, hspan)

    plot!(myp, coordout[:,1], coordout[:,2], label="Coord Barker")

    rwmtest = function(reps,h)

        aout = 0
        esjd = 0
        
        for _ in 1:reps
            x = rand(TDist(nu), d) #Position vector, initialised at x0
            
            fx = logf(x)
            
            v = randn(d)

            #Proposal position, log-density and gradient
            xprime = x + h*v

            fxprime = logf(xprime)
            
            #Compute acceptance probability
            a = fxprime - fx
            
            aout += min(1,exp(a))

            esjd += min(1,exp(a))*norm(x - xprime)^2
        end

        return([aout/reps, esjd/reps])
    end
    
    bigrwmtest = function(reps, hspan)
        n = length(hspan)

        out = zeros(n,2)

        for i in 1:n
            out[i,:] .= rwmtest(reps,hspan[i])
        end

        return out
    end
    
    rwmout = bigrwmtest(reps, hspan)

    plot!(myp, rwmout[:,1], rwmout[:,2], label="RWM")

    
    naivetest = function(reps,h)

        aout = 0
        esjd = 0
        
        for _ in 1:reps
            x = rand(TDist(nu), d)    #Position vector, initialised at x0
            
            fx = logf(x) #Precalculate density at position
            gradx = gradlogf(x) #Precalculate gradient at position

            v = h*randn(d)

            #Flip stepspflip = @. -log1_exp(-v*grad)
            pflip = -log1_exp(-sum(v.*gradx))

            #Random variables
            u = log(rand())

            #Test which variables we flip
            flips = u > pflip

            #Flip appropriate terms
            v *= (-1)^flips

            #Proposal position, log-density and gradient
            xprime = x + v

            fxprime = logf(xprime)
            gradxprime = gradlogf(xprime)

            #Compute acceptance probability
            a = fxprime - fx + sum(log1_exp.(-v .* gradx)) - sum(log1_exp.(v .* gradxprime))
        
            aout += min(1,exp(a))

            esjd += min(1,exp(a))*norm(v)^2
        end

        return([aout/reps, esjd/reps])
    end

    bignaivetest = function(reps, hspan)
        n = length(hspan)

        out = zeros(n,2)

        for i in 1:n
            out[i,:] .= naivetest(reps,hspan[i])
        end

        return out
    end

    naivout = bignaivetest(reps, hspan)

    plot!(myp, naivout[:,1], naivout[:,2], label="Naive Barker")




    #myvar = sum(lighttailsim(N).^2)/N

    mu = zeros(d)
    sigma = sqrt(d)*I(d)

    stereotest = function(reps,h)

        aout = 0
        esjd = 0
        
        for _ in 1:reps
            x = rand(TDist(nu), d)    #Position vector, initialised at x0
            
            z = SPinv(x; sigma = sigma, mu = mu, isinv = false) #Map to the sphere

            #Calculate log-density on the sphere
            densz = logf(x) - d*log(1-z[end])
                    
            #Set up gradient of potential on the sphere
            gradstereo = SPgradlog(gradlogf)
            gradz = gradstereo(z; sigma=sigma, mu=mu).gradz

            #Orthogonalise gradient
            gradz -= sum(z .* gradz)*z

            normgradz = norm(gradz)

            
            v = h*randn(d)

            #Flip steps
            y = BarkerStep(normgradz/sqrt(d)*ones(d),v)
            
            #Calculate rotated position
            g = RotatezN(z,gradz)[1:end-1] #gradient when z rotated to N

            zprime = z + RotatezNinv(z,vcat(Rotategrad1inv(g,y),[0])) #Add on rotated step
            normalize!(zprime) #project down to sphere

            #Calculate gradient and projection at proposal
            (gradprime, xprime) = gradstereo(zprime; sigma=sigma, mu=mu)
            
            #Orthogonalise gradient
            gradprime -= sum(zprime .* gradprime)*zprime

            #Calculate log-density on the sphere
            densprime = logf(xprime) - d*log(1-zprime[end])
            
            #Normalise gradient for reverse Barker step
            normgradprime = norm(gradprime)

            #Calculate reverse step
            gprime = RotatezN(zprime,gradprime)[1:end-1] #gradient when zprime rotated to N

            yprime = Rotategrad1(gprime, RotatezN(zprime, z/sum(z .* zprime) - zprime)[1:end-1]) #Reverse Barker proposal step 

            #Compute acceptance probability
            a = densprime - densz + sum(log1_exp.(-y*normgradz/sqrt(d))) - sum(log1_exp.(-yprime*normgradprime/sqrt(d)))
            
            aout += min(1,exp(a))

            esjd += min(1,exp(a))*norm(x - xprime)^2
        end

        return([aout/reps, esjd/reps])
    end

    bigstereotest = function(reps, hspan)
        n = length(hspan)

        out = zeros(n,2)

        for i in 1:n
            out[i,:] .= stereotest(reps,hspan[i]/sqrt(d))
        end

        return out
    end

    stereoout = bigstereotest(reps, hspan)

    plot!(myp, stereoout[:,1], stereoout[:,2], label=string("Stereo Barker"))
    #plot!(myp, stereoout1[:,1], stereoout1[:,2], label=string("Stereo Barker (suboptimal ",L"$\gamma$",")"), linewidth=3)
    #plot!(myp, stereoout2[:,1], stereoout2[:,2], label=string("Stereo Barker (bad ",L"$\gamma$",")"), linewidth=3)

    #mu .+= 1

    smalatest = function(reps,h)

        aout = 0
        esjd = 0
        
        for _ in 1:reps
            x = rand(TDist(nu), d)    #Position vector, initialised at x0

            z = SPinv(x; sigma = sigma, mu = mu, isinv = false) #Map to the sphere

            #Calculate log-density on the sphere
            fx = logf(x)
                    
            #Set up gradient of potential on the sphere
            gradstereo = SPgradlog(gradlogf)
            gradz = gradstereo(z; sigma=sigma, mu=mu).gradz

            #Orthogonalise gradient
            gradz -= sum(z .* gradz)*z
            
            dz = h*randn(d+1) #Gaussian step
            dz -= sum(z.*dz)*z #Project step onto the tangent plane at z

            zprime = normalize(z + h^2/2*gradz + dz) #New proposed point

            (gradprime, xprime) = gradstereo(zprime; sigma=sigma, mu=mu)
            
            #Orthogonalise gradient
            gradprime -= sum(zprime .* gradprime)*zprime

            fxprime = logf(xprime) #Density at xprime

            zdot = sum(z .* zprime)

            #Compute log-acceptance probability, based on projected density
            a = (-fx + d*log(1 - z[end]) + fxprime - d*log(1 - zprime[end]) 
                - 1/(2h^2)*norm(z/zdot - zprime - h^2/2* gradprime)^2 
                + 1/(2h^2)*norm(zprime/zdot - z - h^2/2* gradz)^2)

            aout += min(1,exp(a))

            esjd += min(1,exp(a))*norm(x - xprime)^2
        end

        return([aout/reps, esjd/reps])
    end

    bigsmalatest = function(reps, hspan)
        n = length(hspan)

        out = zeros(n,2)

        for i in 1:n
            out[i,:] .= smalatest(reps,hspan[i])
        end

        return out
    end

    smalaout = bigsmalatest(reps, hspan)

    plot!(myp, smalaout[:,1], smalaout[:,2], label = "SMALA")
    
    plot!(hspan,smalaout[:,2])

    srwtest = function(reps,h)

        aout = 0
        esjd = 0
        
        for _ in 1:reps
            x = rand(TDist(nu), d)    #Position vector, initialised at x0

            fx = logf(x)

            z = SPinv(x; sigma = sigma, mu = mu, isinv = false) #Map to the sphere

            dz = h*randn(d+1) #Gaussian step
            dz -= sum(z.*dz)*z #Project step onto the tangent plane at z

            zprime = normalize(z + dz) #New proposed point
            xprime = SP(zprime; sigma = sigma, mu = mu) #Project to Euclidean Space

            fxprime = logf(xprime) #Density at xprime

            #Compute log-acceptance probability, based on projected density
            a = -fx + d*log(1 - z[end]) + fxprime - d*log(1 - zprime[end])


            aout += min(1,exp(a))

            esjd += min(1,exp(a))*norm(x - xprime)^2
        end

        return([aout/reps, esjd/reps])
    end

    bigsrwtest = function(reps, hspan)
        n = length(hspan)

        out = zeros(n,2)

        for i in 1:n
            out[i,:] .= srwtest(reps,hspan[i])
        end

        return out
    end

    srwout = bigsrwtest(reps, hspan)

    plot!(myp, srwout[:,1], srwout[:,2], label="SRW")

    plot!(myp, legend=false, legendfontsize=15)
    save("lightESJD.png", myp)

#Rotate Barker Sim Tests

    d = 20
    nu = 20
    b = 1

    #Banana parameters:
    sigma = diagm(vcat(45, sqrt(d)*ones(d-1)))
    mu = vcat(-20,zeros(d-1))

    d > 1 ? x0 = sigma*normalize(randn(d)) + mu : x0 = (sigma*rand([1,-1]))[1]

    #logf = x -> -sum(x.^2)/2
    #logf = x -> -(nu+d)/2*log(nu + sum(x.^2))

    banana(x; b=0) = vcat(x[1] + b*sum(x -> x^2,x[2:end]), x[2:end])
    test = x -> -(nu+d)/2*log(nu + sum(y -> y^2, x))

    logf = x -> test(banana(x; b = b))

    #Set up gradient
    d > 1 ? gradlogf = x -> ForwardDiff.gradient(logf,x) : gradlogf = x -> ForwardDiff.derivative(logf,x)

    #This is here to precalculate the gradient function
    gradlogf(x0)

    N = 100000
    h0 = 3e-2
    epsilon = 2.2e-1
    steps = 300
    
    beta = 1.1
    burnin = floor(Int64, N)
    adaptlength = floor(Int64, N/1000)
    R = 1e6
    r = 1e-3
    forgetrate = 3/4
    hgeom = 10
    
    @time out = SBarkerAdaptive(logf, x0, h0, N, beta, r, R; gradlogf = gradlogf, 
        sigma = sigma, mu = mu, burnin = burnin, adaptlength = burnin, steps = steps, forgetrate = forgetrate, updategamma = false, updateh = true, hgeom = hgeom);

    #1700s

    @time rotateout = RotateBarkerSim(logf, x0, epsilon, N; gradlogf = gradlogf, includefirst = true, steps = steps, printing = false)
    @time coordout = CoordBarkerSim(logf, x0, epsilon, N; gradlogf = gradlogf, includefirst = true, steps = steps, printing = false)
    @time malaout = HMC(logf, gradlogf, x0, N, epsilon, 1; steps = steps);

    save("BananaData.jld", "out", out, "rotateout", rotateout, "coordout", coordout, "malaout", malaout)

    @time slice = SliceAdaptive(logf, x0, N, beta, r, R; 
        sigma = sigma, mu = mu, burnin = burnin, adaptlength = burnin, steps = steps, forgetrate = forgetrate, updategamma = true);
    
    @time HMCout = eff_NUTS(x0, epsilon, logf, N; ∇L = gradlogf, Δ_max = 1000, steps = 1);
    #2400s


    #Plot comparison against the true distribution
    p(x) = 1/sqrt(2pi)*exp(-x^2/2)
    #q(x) = 1/4*gamma(1/4)*exp(-x^4)
    #q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
    b_range = range(-10,10, length=101)
    histogram(malaout.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
    plot!(p, label= "N(0,1)", lw=3)
    #plot!(q, label= "light", lw=3)
    xlabel!("x")
    ylabel!("P(x)")

    histogram(malaout.x[:,1], label="Experimental", normalize=:pdf)
    histogram!(coordout.x[:,1], label="Experimental", normalize=:pdf)
    histogram!(rotateout.x[:,1], label="Experimental", normalize=:pdf)
    

    
    plot(1:steps:N*steps,out.x[:,1], label = "x_1")
    vline!(steps*cumsum(out.times[1:end-1]), label = "Adaptations", lw = 0.5)

    plot(1:steps:N*steps,slice.x[:,1], label = "x_1")
    vline!(steps*cumsum(slice.times[1:end-1]), label = "Adaptations", lw = 0.5)

    plot(1:steps:N*steps,HMCout[:,1], label = "x_1")
   
    plot(1:steps:N*steps,out.z[:,end], label = "z_{d+1}")
    vline!(steps*cumsum(out.times[1:end-1]), label = "Adaptations", lw = 0.5)

    stereobana = sum(out.x[:,2:end], dims = 2)
    rotatebana = sum(rotateout.x[:,2:end], dims = 2)
    coordbana = sum(coordout.x[:,2:end], dims = 2)
    malabana = sum(malaout.x[:,2:end], dims = 2)


    plot(1:steps:N*steps,barkerbana[:,1])

    #histogram2d(slice.x[:,1], slice.x[:,2],normalize=:pdf, bins=(1000,1000))
    histogram2d(slice.x[:,1], slicebana,normalize=:pdf, bins=(1000,1000))
    histogram2d(out.x[:,1], barkerbana[:,1], normalize=:pdf, bins=(1000,1000))

    plot((0:15:450)*steps, autocor(rotateout.x[:,1], (0:15:450)), label = "Rotate")
    hline!([0], label="")
    plot!((0:15:450)*steps,autocor(malaout.x[:,1], (0:15:450)), label = "MALA")
    plot!((0:15:450)*steps,autocor(coordout.x[:,1], (0:15:450)), label = "Coord")
    plot!((0:15:450)*steps,autocor(out.x[:,1], (0:15:450)), label = "Stereo")
    savefig("BananaAutocor.png")
    
    plot((0:1:30)*steps, autocor(rotatebana, (0:1:30)), label = "Rotate")
    hline!([0], label="")
    plot!((0:1:30)*steps,autocor(malabana, 0:1:30), label = "MALA")
    plot!((0:1:30)*steps,autocor(coordbana, (0:1:30)), label = "Coord")
    plot!((0:1:30)*steps,autocor(stereobana, (0:1:30)), label = "Stereo")



# Plots for poster 
    q(y,grad) = sqrt(2/pi)*exp(-y^2/2)/(1 + exp(-y*grad))

    myp = plot()
    
    for grad in [0,2,8,Inf]
        plot!(myp,y -> q(y,grad), label= string(L"$\nabla\log\pi = $", grad), xlims = [-3,3], linewidth=3)
    end
    plot!(myp, legend=:topleft, legendfontsize=16)
    #vline!(myp,[0], line = :black)
    plot(myp)

    save("BarkerProposal.png",myp)

    grad = [0,0]
    q(x,y) = 2/pi*exp(-(y^2+x^2)/2)/(1 + exp(-x*grad[1]))/(1 + exp(-y*grad[2]))
    
    x = range(-3, 3, length=100)
    y = range(-3, 3, length=100)
    z = @. q(x', y)
    myplot=contour(x, y, z, cbar=false, ylims=[-3,3], aspect_ratio=:equal)
    plot!(myplot,[0,0],[-2.5,2.5],arrow=true,color=:black,linewidth=2,label="")
    plot!(myplot,[-1,2.75],[0,0],arrow=true,color=:black,linewidth=2,label="")
    
    save("CoordBarkerProp.png", myplot)

    qrotate(x,y) = 2/pi*exp(-(y^2+x^2)/2)/(1 + exp(-Rotategrad1(grad,[x,y])[1]*norm(grad)/sqrt(2)))/(1 + exp(-Rotategrad1(grad,[x,y])[2]*norm(grad)/sqrt(2)))

    x = range(-1, 3, length=100)
    y = range(-3, 3, length=100)
    z = @. qrotate(x', y)
    myplot=contour(x, y, z, cbar=false,ylims=[-3,3], aspect_ratio=:equal)
    plot!(myplot,[-1,2.5],[-1,2.5],arrow=true,color=:black,linewidth=2,label="")
    plot!(myplot,[-1,2.5],[1,-2.5],arrow=true,color=:black,linewidth=2,label="")
    
    save("RotateBarkerProp.png",myplot)

#Logistic Regression Tests
    d = 50
    nu = 3
    p = 1000

    s = 10

    #True parameters
    alphatrue = 1
    betaindic = rand(Bernoulli(0.3), d-1)
    betatrue = s*rand(TDist(nu), d-1) .* betaindic

    #Stereo parameters:
    mu = vcat(alphatrue, betatrue)
    sigma = I(d)* s .+ vcat(0, betaindic) * s^2

    data = randn(p,d-1)

    probs = 1 ./(1 .+ exp.(-alphatrue .- data*betatrue))
    obs = rand.(Bernoulli.(probs))

    plot(obs); scatter!(probs)
    plot(vcat(alphatrue, betatrue))

    plot(obs.*log.(probs) + (1 .- obs).*log.(1 .-probs))


    logf = function(x)
        alpha = x[1]
        beta = x[2:end]

        logprior = -alpha^2/2s^2 -(nu+d-1)/2*log(nu + sum(y -> y^2, beta)/s^2)

        logprobsplus = -log1_exp.(-alpha .- data*beta)
        logprobsminus = -log1_exp.(alpha .+ data*beta)

        loglikelihood = sum(obs.*logprobsplus + (1 .- obs).*logprobsminus)

        return logprior + loglikelihood
    end

    #Set up gradient
    d > 1 ? gradlogf = x -> ForwardDiff.gradient(logf,x) : gradlogf = x -> ForwardDiff.derivative(logf,x)

    #Initial position
    d > 1 ? x0 = sigma*normalize(randn(d)) + mu : x0 = (sigma*rand([1,-1]))[1]

    #This is here to precalculate the gradient function
    gradlogf(x0)

    N = 1000000
    h0 = 6.5e-2
    epsilon = 2.2e-1
    steps = 10
    
    beta = 1.1
    burnin = floor(Int64, N/2000)
    adaptlength = floor(Int64, N/2000)
    R = 1e6
    r = 1e-3
    forgetrate = 3/4
    hgeom = 10
    
    #Parameter setup
        @time slice = SliceAdaptive(logf, x0, N, beta, r, R; 
            sigma = sigma, mu = mu, burnin = burnin, adaptlength = burnin, steps = steps, forgetrate = forgetrate, updategamma = true);
        
        plot(slice.mu[end,:])
        plot!(vcat(alphatrue, betatrue))

        plot(diag(slice.sigma[end]))
        plot!(abs.(vcat(alphatrue, betatrue)))

        sigma = slice.sigma[end]
        mu = slice.mu[end,:]

        
        save("parameters.jld", "alphatrue", alphatrue, "betatrue", betatrue, "betaindic", betaindic, "data", data, "obs", obs, "sigma", sigma, "mu", mu)
    
    #alphatrue = load("parameters.jld")["alphatrue"]; betatrue = load("parameters.jld")["betatrue"]; betaindic = load("parameters.jld")["betaindic"]; data = load("parameters.jld")["data"]; obs = load("parameters.jld")["obs"]; sigma = load("parameters.jld")["sigma"]; mu = load("parameters.jld")["mu"]


    #Initial position
    d > 1 ? x0 = sigma*normalize(randn(d)) + mu : x0 = (sigma*rand([1,-1]))[1]
    #This is here to precalculate the gradient function
    gradlogf(x0)
    
    
    @time out = SBarkerAdaptive(logf, x0, h0, N, beta, r, R; gradlogf = gradlogf, 
        sigma = sigma, mu = mu, burnin = burnin, adaptlength = burnin, steps = steps, forgetrate = forgetrate, updategamma = false, updateh = true, hgeom = hgeom);

    save("out.jld", "out", out)

    @time rotateout = RotateBarkerAdaptive(logf, x0, epsilon, N, beta, r, R; gradlogf = gradlogf, 
        burnin = burnin, adaptlength = burnin, steps = steps, forgetrate = forgetrate, updateh = true, hgeom = hgeom);

    save("rotateout.jld", "rotateout", rotateout)
    
    @time coordout = CoordBarkerAdaptive(logf, x0, epsilon, N, beta, r, R; gradlogf = gradlogf, 
        burnin = burnin, adaptlength = burnin, steps = steps, forgetrate = forgetrate, updateh = true, hgeom = hgeom);

    save("coordout.jld", "coordout", coordout)

    @time malaout = MalaAdaptive(logf, x0, epsilon, N, beta, r, R; gradlogf = gradlogf, 
        burnin = burnin, adaptlength = burnin, steps = steps, forgetrate = forgetrate, updateh = true, hgeom = hgeom);

    save("malaout.jld", "malaout", malaout)

    @time rotateout = RotateBarkerSim(logf, x0, epsilon, N; gradlogf = gradlogf, includefirst = true, steps = steps, printing = false)
    @time coordout = CoordBarkerSim(logf, x0, epsilon, N; gradlogf = gradlogf, includefirst = true, steps = steps, printing = false)
    @time malaout = HMC(logf, gradlogf, x0, N, epsilon, 1; steps = steps);

    out = load("out.jld")["out"]
    rotateout = load("rotateout.jld")["rotateout"]
    coordout = load("coordout.jld")["coordout"]
    malaout = load("malaout.jld")["malaout"]
    


    #Plot comparison against the true distribution
    p(x) = 1/sqrt(2pi)*exp(-x^2/2)
    #q(x) = 1/4*gamma(1/4)*exp(-x^4)
    #q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
    b_range = range(-10,10, length=101)
    histogram(malaout.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
    plot!(p, label= "N(0,1)", lw=3)
    #plot!(q, label= "light", lw=3)
    xlabel!("x")
    ylabel!("P(x)")

    histogram(malaout.x[:,1], label="Experimental", normalize=:pdf)
    histogram!(coordout.x[:,1], label="Experimental", normalize=:pdf)
    histogram!(rotateout.x[:,1], label="Experimental", normalize=:pdf)
    

    
    plot(1:steps:N*steps,out.x[:,15], label = "Stereo")
    plot!(1:steps:N*steps,malaout.x[:,15], label = "MALA")
    plot!(1:steps:N*steps,rotateout.x[:,15], label = "Rotate")
    plot!(1:steps:N*steps,coordout.x[:,15], label = "Coord")
    
    savefig("LogisticBetaTraceplot.png")
    
    plot(1:steps:N*steps,out.x[:,1], label = "Stereo")
    plot!(1:steps:N*steps,malaout.x[:,1], label = "MALA")
    plot!(1:steps:N*steps,rotateout.x[:,1], label = "Rotate")
    plot!(1:steps:N*steps,coordout.x[:,1], label = "Coord")
    
    savefig("LogisticAlphaTraceplot.png")
    
    plot(1:steps:N*steps,outnorm, label = "x_1")
    plot!(1:steps:N*steps,malanorm, label = "x_1")
    plot!(1:steps:N*steps,rotatenorm, label = "x_1")
    plot!(1:steps:N*steps,coordnorm, label = "x_1")
    

    plot((0:1:40)*steps,autocor(out.x[:,1], (0:1:40)), label = "Stereo")
    plot!((0:1:40)*steps,autocor(malaout.x[:,1], (0:1:40)), label = "MALA")
    plot!((0:1:40)*steps, autocor(rotateout.x[:,1], (0:1:40)), label = "Rotate")
    plot!((0:1:40)*steps,autocor(coordout.x[:,1], (0:1:40)), label = "Coord")
    hline!([0], label="")
    savefig("LogisticAutocorAlpha.png")
    
    
    plot((0:20:6000)*steps,autocor(out.x[:,15], (0:20:6000)), label = "Stereo")
    plot!((0:200:6000)*steps,autocor(malaout.x[:,15], (0:200:6000)), label = "MALA")
    plot!((0:200:6000)*steps, autocor(rotateout.x[:,15], (0:200:6000)), label = "Rotate")
    plot!((0:200:6000)*steps,autocor(coordout.x[:,15], (0:200:6000)), label = "Coord")
    hline!([0], label="")
    savefig("LogisticAutocorBeta.png")
    
    outnorm = sum(x -> x^2, out.x, dims=2)
    rotatenorm = sum(x -> x^2, rotateout.x, dims=2)
    coordnorm = sum(x -> x^2, coordout.x, dims=2)
    malanorm = sum(x -> x^2, malaout.x, dims=2)

    
    plot((0:300:9000)*steps, autocor(rotatenorm, (0:300:9000)), label = "Rotate")
    hline!([0], label="")
    plot!((0:300:9000)*steps,autocor(malanorm, (0:300:9000)), label = "MALA")
    plot!((0:300:9000)*steps,autocor(coordnorm, (0:300:9000)), label = "Coord")
    plot!((0:300:9000)*steps,autocor(outnorm, (0:300:9000)), label = "Stereo")
    