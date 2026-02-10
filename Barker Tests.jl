### Tests for Barker MCMC 

#Import libraries for testing
    include("Adaptive SBPS.jl")
    include("Hamiltonian MC.jl")
    include("Adaptive SRW.jl")
    include("SHMC.jl")
    include("Adaptive Slice.jl")
    include("SBarker.jl")

    using ForwardDiff
    using Plots
    using SpecialFunctions
    using StatsBase
    using JLD
    using LaTeXStrings

#Tests
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

