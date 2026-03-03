### Stereographic Barker Code ###

#Import Stereographic Projection stuff, and the Random library
include("Stereographic Projection.jl")
using Random

#Rotation operator which moves z to N, applied to v
RotatezN = function (z,v)
    #Ensure we are on the sphere, length(z)==length(v)
    abs(sum(z.^2) - 1) >= 1e-12 && error("norm(z) != 1") 
    
    abs(sum(v.^2) - 1) >= 1e-12 && error("|v| != 1") 
    
    length(z) != length(v) && error("length(z) != length(v)") 

    #Include South Pole case
    z[end] == -1 && return(-v)

    #Prepare output
    out = zeros(length(z))

    #Precalculate dot product
    zdotv = sum(z .* v)

    #Calculate rotation
    out[1:end-1] = v[1:end-1] - (v[end] + zdotv)/(1+z[end])*z[1:end-1]
    out[end] = zdotv

    #Return output
    return(out)
end

#Rotation operator moves N to z, applied to v
RotatezNinv = function (z,v)
    #Ensure we are on the sphere, length(z)==length(v)
    abs(sum(z.^2) - 1) >= 1e-12 && error("norm(z) != 1") 
    
    abs(sum(v.^2) - 1) >= 1e-12 && error("|v| != 1") 
    
    length(z) != length(v) && error("length(z) != length(v)") 

    #Include South Pole case
    z[end] == -1 && return(-v)

    #Prepare output
    out = zeros(length(z))

    #Precalculate dot product excluding last component
    zdotv = sum(z[1:end-1] .* v[1:end-1])

    #Calculate rotation
    out[1:end-1] = v[1:end-1] + (v[end] + z[end]*v[end]- zdotv)/(1+z[end])*z[1:end-1]
    out[end] = z[end]*v[end]- zdotv

    #Return output
    return(out)
end

#Rotation operator which moves ones(d) to grad, applied to y
#If grad has a latitude of 0 (e.g. after using RotatezN), make sure to truncate before inputing
Rotategrad1 = function (grad,y)
    #Ensure grad and y have the same length
    length(grad) != length(y) && error("length(grad) != length(y)")

    #Dimension
    d = length(grad)

    #Normalise gradient
    gradnorm = normalize(grad)

    #Precalculate relevant sums
    grad1 = sum(gradnorm)/sqrt(d)
    y1 = sum(y)/sqrt(d)
    graddoty = sum(gradnorm.*y)

    #Include -1 case
    grad1 == -1 && return(-y)

    #Return rotation
    return(y .+ ((graddoty*(1+ 2grad1) - y1)*ones(d)/sqrt(d) .- (y1 + graddoty)*gradnorm)/(1 + grad1))
end

#Rotation operator which moves grad to ones(d), applied to y
#If grad has a latitude of 0 (e.g. after using RotatezN), make sure to truncate before inputing
Rotategrad1inv = function (grad,y)
    #Ensure grad and y have the same length
    length(grad) != length(y) && error("length(grad) != length(y)")

    #Dimension
    d = length(grad)

    #Normalise gradient
    gradnorm = normalize(grad)

    #Precalculate relevant sums
    grad1 = sum(gradnorm)/sqrt(d)
    y1 = sum(y)/sqrt(d)
    graddoty = sum(gradnorm.*y)

    #Include -1 case
    grad1 == -1 && return(-y)

    #Return rotation
    return(y .+ ((y1*(1+ 2grad1) - graddoty)*gradnorm .- (graddoty + y1)*ones(d)/sqrt(d))/(1 + grad1))
end

#Barker step flipping against directional gradient
#For rotate Barker, gradi = || ∇ log π ||/sqrt(d)
#For coordinate Barker, gradi = d log π/dx_i
BarkerStep = function (grad,v)
    #Initialise Dimension
    d = length(v)

    d != length(grad) && error("Dimension error: length(v) != length(grad)")

    #Prepare output
    y = copy(v)

    #log-Probability of flipping
    pflip = @. -log(1 + exp(-v*grad))

    #Random variables
    u = log.(rand(d))

    #Test which variables we flip
    flips = u .> pflip

    #Flip appropriate terms
    @. y *= (-1).^flips

    return(y)
end

#Rotate Barker Simulator
RotateBarkerSim = function (logf, gradlogf, x0, h, N; includefirst = true, steps = 1, printing = false)
    d = length(x0) #The dimension
    #d < 2 && error("Still working on d=1 case")

    #Prepare output
    xout = zeros(N,d)

    #Slightly convoluted method for not storing the initial value WITHOUT allocating memory for an entirely new matrix
    if includefirst
        #If we want to include the first value, initialise the outputs
        indexes = 2:N
        xout[1,:] .= x0
    else
        #If we don't, start the indexes to be inputted at 1
        indexes = 1:N
    end

    x = x0 #Position vector, initialised at x0
    
    fx = logf(x) #Precalculate density at position
    d > 1 ? gradx = gradlogf(x) : gradx = gradlogf(x[1]) #Precalculate gradient at position
    normgradx = norm(gradx) #Precalculate norm(gradient) at position
    
    aout = 0 # Track acceptance rate
    
    for n in indexes
        #Print iteration number
        printing && print("\rStep number: $n")

        #We only sample one point after several steps
        for _ in 1:steps
            #Initialise step
            v = h*randn(d)

            #Flip steps
            y = BarkerStep(normgradx/sqrt(d)*ones(d),v)
            
            #Proposal position, log-density and gradient
            xprime = x .+ Rotategrad1inv(gradx, y)

            fxprime = logf(xprime)
            d > 1 ? gradxprime = gradlogf(xprime) : gradxprime = gradlogf(xprime[1])
            normgradxprime = norm(gradxprime)

            #Calculate reverse step
            yprime = Rotategrad1(gradxprime,x.-xprime)

            #Compute acceptance probability
            a = fxprime - fx + sum(log.(1 .+ exp.(-y*normgradx/sqrt(d)))) - sum(log.(1 .+ exp.(-yprime*normgradxprime/sqrt(d))))

            u = log(rand(Float64)) #Simulate from uniform to accept/reject

            if u < a #Accept proposal
                #Update position, density, gradient
                x = xprime 
                fx = fxprime
                gradx = gradxprime
                normgradx = norm(gradx)

                aout += 1/(N*steps-1)
            end
        end

        #Add to output
        xout[n,:] .= x
    end
    println()

    return (x = xout, a = aout)
end