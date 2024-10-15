#Import LinearAlgebra library
using LinearAlgebra

## Stereographic Projection ##

#Stereographic Projection from z to x
SP = function (z; sigma = sqrt(length(z)-1)I(length(z)-1), mu = zeros(length(z)-1))
    #Check that norm(z) == 1
    abs(sum(z.^2) -1) >= 1e-12 && error("norm(z) != 1")

    #Warning if z == North Pole
    #z[end] - 1 == 0 && (println("Warning: x = Inf"); return Inf*ones(length(z)-1))

    sigma*z[1:end-1]/(1- z[end]) .+ mu
end

#Stereographic Projection from x to z
SPinv = function (x; sigma = sqrt(length(x))I(length(x)), mu = zeros(length(x)), isinv = false)
    isinv ? y = sigma*(x.-mu) : y = inv(sigma)*(x.-mu)

    ynorm = sum(y.^2)

    z = Vector{Float64}(undef, length(x)+1)

    z[1:end-1] = 2y/(ynorm + 1)
    z[end] = (ynorm -1)/(ynorm +1)

    return z
end

#Pdf of z = SPinv(x), where x has pdf f
SPdensity = function(f; logdens = false)
    #Output of SPdensity will be a function giving the projected density
    if logdens
        #pi_S outputs f(x) + log(Jacobian) of SP if we ask for log-density
        pi_S = function(z; sigma = sqrt(length(z)-1)I(length(z)-1), mu = zeros(length(z)-1))
            #Transform from sphere to Euclidean space
            x = SP(z; sigma, mu)
    
            #Calculate density on the sphere
            return(piz = f(x)-(length(z)-1)log(1-z[end]), x = x)
        end
    else
        #pi_S outputs f(x)* Jacobian of SP if we ask for density
        pi_S = function(z; sigma = sqrt(length(z)-1)I(length(z)-1), mu = zeros(length(z)-1))
            #Transform from sphere to Euclidean space
            x = SP(z; sigma, mu)
    
            #Calculate density on the sphere
            return(piz = f(x)*(1-z[end])^-(length(z)-1), x = x)
        end
    end

    return pi_S
end

#Gradient of logf(z) at z if x has pdf f
SPgradlog = function (gradlogf)
    gradlogz = function (z; sigma = sqrt(length(z)-1)I(length(z)-1), mu = zeros(length(z)-1))
        #Project to Euclidean space
        x = SP(z; sigma, mu)
        d = length(x)

        #Perform the rate calculation
        #Need dimension check for d=1 case
        d > 1 ? gradx = gradlogf(x) : gradx = gradlogf(x[1])

        #Calculate gradient of density on sphere
        return(gradz = vcat(sigma*gradx, d + sum((x.-mu).*gradx))/(1-z[end]), x = x)
    end
end

## SBPS Setup ##

#SBPS Deterministic path of length t, discretised with time step delta
SBPSDetPath = function(z,v,t,delta; t0 = 0)
    #Ensure we are on the sphere, length(z)==length(v) and z.v=0
    abs(sum(z.^2) - 1) >= 1e-12 && error("norm(z) != 1") 
    
    abs(sum(v.^2) - 1) >= 1e-12 && error("|v| != 1") 
    
    length(z) != length(v) && error("length(z) != length(v)") 
    
    abs(sum(z.*v)) >= 1e-12 && error("z.v != 0")
    
    #If we have t0 > t, the path is so short that we do not observe any points along this piece
    t0 > t && error("no observed points")

    #Dimension (technically d+1 is the dimension, but this is easier for coding)
    d = length(z)
    
    #Skeleton path times
    #t0 is present because the skeleton path will not exactly start at event times
    times = collect(t0:delta:t)
    n = length(times)
    
    zout = zeros(n,d)
    vout = zeros(n,d)
    
    for i = 1:d 
        @. zout[:,i] = z[i]cos(times) + v[i]sin(times)
        @. vout[:,i] = v[i]cos(times) - z[i]sin(times)
    end

    return (z = zout, v = vout)
end

#SBPS Refreshment update for velocity
SBPSRefresh = function(z)
    #Ensure we are on the sphere, length(z)==length(v) and z.v=0
    abs(sum(z.^2) - 1) >= 1e-12 && error("norm(z) != 1")
    
    #To sample uniformly from {z}^perp, we sample v from a Normal distribution,
    #then return (v - (z.v)z)/|| v - (z.v)z ||
    v = randn(length(z))
    
    normalize(v - sum(z.*v)z)
end

#SBPS Bounce update for velocity, given gradient of spherical density
SBPSBounce = function(z,v,grad)
    #Ensure we are on the sphere, length(z)==length(v) and z.v=0
    abs(sum(z.^2) - 1) >= 1e-12 && error("norm(z) != 1") 
    
    abs(sum(v.^2) - 1) >= 1e-12 && error("|v| != 1") 
    
    length(z) != length(v) && error("length(z) != length(v)") 
    
    abs(sum(z.*v)) >= 1e-12 && error("z.v != 0")
    
    #Evaluate the normalised gradient tangent to z
    gradtangent = normalize(grad - sum(z.*grad)z)

    v - 2sum(v.*gradtangent)gradtangent
end

#WARNING: SBPSRate outputs the gradient and dot product for the SBPS event rate.
#In particular, you need to take -dot product for the true rate
SBPSRate = function (gradlogf) #Note this function requires the ∇log(f) already calculated
    #Output is the function lambda_S
    lambda_S = function(t,z0,v0; sigma = sqrt(length(z)-1)I(length(z)-1), mu = zeros(length(z)-1))
        #Move t time units forward from z0, v0
        (z,v) = (z0*cos(t) + v0*sin(t), v0*cos(t) - z0*sin(t))

        #Project to Euclidean space
        x = SP(z; sigma, mu)
        d = length(x)

        #Perform the rate calculation
        #Need dimension check for d=1 case
        d > 1 ? xgrad = gradlogf(x) : xgrad = gradlogf(x[1])

        #Calculate gradient of density on sphere
        zgrad = vcat(sigma*xgrad, d + sum((x.-mu).*xgrad))/(1-z[end])

        #Return the rate and the gradient
        return (rate = sum(v.*zgrad), grad = zgrad)
    end

    return lambda_S
end

#Derivative of SBPS rate
#Again, need to take -derivative for the derivative of true rate
SBPSRateDeriv = function (hessianlogf,gradlogf)
    #Output is the function Dlambda
    lambdaderiv = function (t,z0,v0; sigma = sqrt(length(z)-1)I(length(z)-1), mu = zeros(length(z)-1))
        #Move t time units forward from z0, v0
        (z,v) = (z0*cos(t) + v0*sin(t), v0*cos(t) - z0*sin(t))

        #Project to Euclidean space
        x = SP(z; sigma, mu)
        d = length(x)

        #Perform the rate calculation
        #Need dimension check for d=1 case
        d > 1 ? xgrad = gradlogf(x) : xgrad = gradlogf(x[1])

        #Perform the hessian calculation
        d > 1 ? hessx = hessianlogf(x) : hessx = hessianlogf(x[1])

        #Calculate gradient of density on sphere
        zgrad = vcat(sigma*xgrad, d + sum((x.-mu).*xgrad))/(1-z[end])

        #Calculate dx/dt
        dxdt = sigma*(v[1:end-1] + v[end]*z[1:end-1]/(1-z[end]))/(1-z[end])

        #Return the rate, the gradient and the derivative of the rate
        return (rate = sum(v.*zgrad), grad = zgrad, 
        derivative = -sum(z.*zgrad) + sum(v.*vcat(sigma*hessx*dxdt, v[end]*sum((x.-mu).*xgrad)/(1-z[end])^2 + sum(dxdt.*xgrad)/(1-z[end]) + sum((x.-mu).*(hessx*dxdt))/(1-z[end]))))
    end
end

#Simulate from a Poisson process with a step function rate
#Takes a sequence of levels and interval lengths
#Be aware, function mutates lambda and t
PoissonStepSim = function (lambda,t; tau0 = 0)

    #Ensure we have as many rates as windows
    length(lambda) != length(t) && error("Mismatch between number of rates and windows")

    #Simulate our Exp(1) random variable
    e = randexp()

    #Event time value
    tau = 0

    #Indicator that no event has occurred
    noevent = true

    #Need to adjust the windows to start the simulation at time tau0
    while length(t) > 0
        j = t[1]

        #if tau0 is larger than the first window, remove first window and reduce tau0
        if tau0 >= j
            tau0 -= j
            popfirst!(t)
            popfirst!(lambda)
        else
            #Otherwise, reduce length of first window to remove the first tau0 time
            t[1] -= tau0
            break
        end
    end

    N = length(t)
    i = 0

    while noevent && i < N
        #Increment window
        i += 1

        tmp = lambda[i]*t[i]

        #Check to see whether there is an event in the current window
        if e >= tmp
            #If not, move on to next window
            e -= tmp
        else
            #Set event time within current window
            tau = sum(t[1:i-1]) + e/lambda[i]
            noevent = false
        end
    end

    #If no event occurred before end of windows, return Inf
    #Otherwise, return the event time and the bound at that time
    noevent ? (return (time = Inf, bound = Inf)) : return (time = tau, bound = lambda[i])
end