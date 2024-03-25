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
    
            f(x)-(length(z)-1)log(1-z[end])
        end
    else
        #pi_S outputs f(x)* Jacobian of SP if we ask for density
        pi_S = function(z; sigma = sqrt(length(z)-1)I(length(z)-1), mu = zeros(length(z)-1))
            #Transform from sphere to Euclidean space
            x = SP(z; sigma, mu)
    
            f(x)*(1-z[end])^-(length(z)-1)
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
        d > 1 ? xgrad = gradlogf(x) : xgrad = gradlogf(x[1])

        #Calculate gradient of density on sphere
        zgrad = vcat(sigma*xgrad, d + sum((x.-mu).*xgrad))/(1-z[end])

        return(zgrad)
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
