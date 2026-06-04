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

    N = 100000
    d = 100
    l = 1
    h = l*d^(-1/6)

    x0 = randn(d) *1e2

    reps = 10000
    hspan = 10 .^(-4:0.02:0.5)
    nu = 100

    mu = zeros(d)
    sigma = sqrt(d)*I(d)

    logf = x -> -sum(x.^2)/2
    #logf = x -> -(nu+d)/2*log(nu + sum(x.^2))
    #logf = x -> -sum(x.^4)

    #Set up gradient
    d > 1 ? gradlogf = x -> ForwardDiff.gradient(logf,x) : gradlogf = x -> ForwardDiff.derivative(logf,x)
    #gradlogf = x -> -x
    #gradlogf = x -> -(nu+d)*x/(nu + sum(x.^2))

    #This is here to precalculate the gradient function
    gradlogf(x0)
    log1_exp = function(x)
         if x < -10 
            return exp(x)
        elseif x > 10
            return x
        else
            return log(1 + exp(x))
        end 
    end

    out = SMALA(logf, x0, h, N; gradlogf = gradlogf, sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)), includefirst = true, steps = 1, printing = false)
    out2 = StereoBarkerSim(logf, x0, h, N; gradlogf = gradlogf, sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)), includefirst = true, steps = 1, printing = false)


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
            x = randn(d) #Position vector, initialised at x0
            #x = lighttailsim(d)
            #x = rand(MvTDist(nu, diagm(ones(d))))

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
    plot!(myp, rotateout[:,1], rotateout[:,2], label="Rotate Barker", linewidth=3)


    malatest = function(reps,h)

        aout = 0
        esjd = 0
        
        for _ in 1:reps
            x = randn(d) #Position vector, initialised at x0
            #x = lighttailsim(d)
            #x = rand(MvTDist(nu, diagm(ones(d))))

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

    plot!(myp, malaout[:,1], malaout[:,2], label="MALA", linewidth=3)
    

    coordtest = function(reps,h)

        aout = 0
        esjd = 0
        
        for _ in 1:reps
            x = randn(d) #Position vector, initialised at x0
            #x = lighttailsim(d)
            #x = rand(MvTDist(nu, diagm(ones(d))))
            
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
            a = fxprime - fx + sum(log.(1 .+ exp.(-y .* gradx))) - sum(log.(1 .+ exp.(y .* gradxprime)))
        
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

    plot!(myp, coordout[:,1], coordout[:,2], label="Coord Barker", linewidth=3)
    plot!(myp, legend=false, legendfontsize=15)
    

    rwmtest = function(reps,h)

        aout = 0
        esjd = 0
        
        for _ in 1:reps
            x = randn(d) #Position vector, initialised at x0
            ##x = lighttailsim(d)
            #x = rand(MvTDist(nu, diagm(ones(d))))
            
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

    myvar = sum(lighttailsim(N).^2)/N

    mu = zeros(d)
    sigma = sqrt(d)*I(d)

    stereotest = function(reps,h)

        aout = 0
        esjd = 0
        
        for _ in 1:reps
            x = randn(d) #Position vector, initialised at x0
            #x = lighttailsim(d)
            #x = randn(d)*1e1
            #x = rand(MvTDist(nu, diagm(ones(d))))
            
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
            a = densprime - densz + sum(log.(1 .+ exp.(-y*normgradz/sqrt(d)))) - sum(log.(1 .+ exp.(-yprime*normgradprime/sqrt(d))))
            
            aout += min(1,exp(a))

            esjd += min(1,exp(a))*norm(x - xprime)^2
        end

        return([aout/reps, esjd/reps])
    end

    bigstereotest = function(reps, hspan)
        n = length(hspan)

        out = zeros(n,2)

        for i in 1:n
            out[i,:] .= stereotest(reps,hspan[i])
        end

        return out
    end

    stereoout = bigstereotest(reps, hspan)

    plot!(myp, stereoout[:,1], stereoout[:,2], label=string("Stereo Barker"), linewidth=3)
    plot!(myp, stereoout1[:,1], stereoout1[:,2], label=string("Stereo Barker (suboptimal ",L"$\gamma$",")"), linewidth=3)
    plot!(myp, stereoout2[:,1], stereoout2[:,2], label=string("Stereo Barker (bad ",L"$\gamma$",")"), linewidth=3)

    #mu .+= 1

    smalatest = function(reps,h)

        aout = 0
        esjd = 0
        
        for _ in 1:reps
            x = randn(d) #Position vector, initialised at x0
            ##x = lighttailsim(d)
            #x = randn(d)*1e2
            #x = rand(MvTDist(nu, diagm(ones(d))))
            
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
            x = randn(d) #Position vector, initialised at x0
            #x = rand(MvTDist(nu, diagm(ones(d))))
            
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
    save("GaussianESJD.png", myp)

#Rotate Barker Sim Tests

    d = 200
    nu = 2

    sigma = sqrt(d)I(d)
    mu = zeros(d) .+ 1e3

    d > 1 ? x0 = sigma*normalize(randn(d)) + mu : x0 = (sigma*rand([1,-1]))[1]

    #banana(x; b=0) = vcat(x[1] + b*x[2]^2,x[2:end])

    #logf = x -> -sum(x.^2)/2
    logf = x -> -(nu+d)/2*log(nu + sum(x.^2))

    #b=0

    #f = x -> test(banana(x; b=b))
    #f = x -> test(banana(x; b=b))
    #f = test
    #f = x -> -sum(x.^2)/2

    #Set up gradient
    d > 1 ? gradlogf = x -> ForwardDiff.gradient(logf,x) : gradlogf = x -> ForwardDiff.derivative(logf,x)

    #This is here to precalculate the gradient function
    gradlogf(x0)

    N = 150000
    h0 = 0.1d^-1
    steps = 50
    
    beta = 1.1
    burnin = N/2000
    adaptlength = N/2000
    R = 1e6
    r = 1e-3
    forgetrate = 3/4
    
    @time out = SBarkerAdaptive(logf, x0, h0, N, beta, r, R; gradlogf = gradlogf, 
        sigma = sigma, mu = mu, burnin = burnin, adaptlength = burnin, steps = steps, forgetrate = forgetrate, updategamma = true, updateh = true, hgeom = 1);

    #@time out = RotateBarkerSim(logf, x0, h, N; gradlogf = gradlogf, includefirst = true, steps = 1, printing = false)

    
    #Plot comparison against the true distribution
    p(x) = 1/sqrt(2pi)*exp(-x^2/2)
    #q(x) = 1/4*gamma(1/4)*exp(-x^4)
    q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
    b_range = range(-10,10, length=101)

    histogram(out.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
    plot!(p, label= "N(0,1)", lw=3)
    plot!(q, label= "light", lw=3)
    xlabel!("x")
    ylabel!("P(x)")

    
    plot(1:steps:N*steps,out.x[:,1], label = "x_1")
    vline!(steps*cumsum(out.times[1:end-1]), label = "Adaptations", lw = 0.5)

    plot(1:steps:N*steps,out.z[:,end], label = "z_{d+1}")
    vline!(steps*cumsum(out.times[1:end-1]), label = "Adaptations", lw = 0.5)


    plot(out.a)

    plot(map(x -> sum(x -> x^2, x), eachrow(out.mu)))
    plot(map(x -> sum(x -> x^2, eigen(x - sqrt(d)I(d)).values), out.sigma))


    #Autocorrelation test
    p = plot()
    sigma = sqrt(d)I(d)
    mu = zeros(d)
    
    d > 1 ? x0 = sigma*normalize(randn(d)) + mu : x0 = (sigma*rand([1,-1]))[1]

    @time out = SBarkerAdaptive(logf, x0, h0, N, beta, r, R; gradlogf = gradlogf, 
        sigma = sigma, mu = mu, burnin = burnin, adaptlength = burnin, steps = steps, forgetrate = forgetrate, updategamma = false, updateh = true, hgeom = 1);
    plot!(p,(0:1:1000)*N, autocor(sum(out.x.^2, dims=2), 0:1:1000), label = "Inf")
    #plot!(p, (0:1:1200)*stepssrw, autocor(sum(out.x.^2, dims=2), 0:1:1200), label = "∞")
    
    hbark = zeros(6)
    #cost[1] = sum(out.Nevals)/T
    hbark[1] = out.h[end]
    i=2
 
    plot(p)


    out=StereoBarkerSim(logf, x0, h0, N; gradlogf = missing, sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)), includefirst = true, steps = 1, printing = false)

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
