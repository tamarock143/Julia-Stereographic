#Import Stereographic Projection stuff, and the Random library
include("Stereographic Projection.jl")
using Random

#We simulate an Stereographic Projection Sampler path targeting the disribtuion f
#To more easily differentiate between SPS and SBPS, we dub this algorithm the Stereographic Random Walk algorithm
SRWSimulator = function(logf, x0, h2, N; sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)), includefirst = true)
    
    z = SPinv(x0; sigma = sigma, mu = mu, isinv = false) #Map to the sphere

    d = length(x0) #The dimension

    #Prepare output
    xout = zeros(N,d)
    zout = zeros(N,d+1)

    #Slightly convoluted method for not storing the initial value WITHOUT allocating memory for an entirely new matrix
    if includefirst
        #If we want to include the first value, initialise the outputs
        indexes = 2:N
        xout[1,:] .= x0
        zout[1,:] .= z
    else
        #If we don't, start the indexes to be inputted at 1
        indexes = 1:N
    end

    x = x0 #Position vector, initialised at x0

    fx = logf(x) #Precalculate density at position

    aout = 0 # Track acceptance rate
    
    for n in indexes
        #Print iteration number
        print("\rStep number: $n")

        dz = h2*randn(d+1) #Gaussian step
        dz -= sum(z.*dz)*z #Project step onto the tangent plane at z

        zprime = normalize(z + dz) #New proposed point
        xprime = SP(zprime; sigma = sigma, mu = mu) #Project to Euclidean Space

        fxprime = logf(xprime) #Density at xprime

        #Compute log-acceptance probability, based on projected density
        a = -fx + d*log(1 - z[end]) + fxprime - d*log(1 - zprime[end])

        u = log(rand(Float64)) #Simulate from uniform to accept/reject

        if u < a #Accept proposal
            #Update position in both Euclidean and Stereographic space
            (x, z, fx) = (xprime, zprime, fxprime) 

            #Keep track of average acceptance probability
            aout += 1/(N - includefirst)
        end

        #Add to output
        xout[n,:] .= x
        zout[n,:] .= z
    end
    println()

    return (x = xout, z = zout, a = aout)
end