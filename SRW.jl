#Import Stereographic Projection stuff, and the Random library
include("Stereographic Projection.jl")
using Random

#We simulate an Stereographic Projection Sampler path targeting the disribtuion exp(f)
#To more easily differentiate between SPS and SBPS, we dub this algorithm the Stereographic Random Walk algorithm
SRWSimulator = function(logf, x0, h2, N; sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)))
    
    z = SPinv(x0; sigma = sigma, mu = mu, isinv = false) #Map to the sphere

    d = length(x0) #The dimension

    #Prepare output
    xout = zeros(N,d)
    zout = zeros(N,d+1)
    xout[1,:] .= x0
    zout[1,:] .= z

    x = x0 #Position vector, initialised at x0

    aout = 0 # Track acceptance rate
    
    for n in 2:N
        #Print iteration number
        print("\rStep number: $n")

        dz = h2*randn(d+1) #Gaussian step
        dz -= sum(z.*dz)*z #Project step onto the tangent plane at z

        zprime = normalize(z + dz) #New proposed point
        xprime = SP(zprime; sigma = sigma, mu = mu) #Project to Euclidean Space

        #Compute acceptance probability, based on projected density
        a = -logf(x) + d*log(1 - z[end]) + logf(xprime) - d*log(1 - zprime[end])

        u = log(rand(Float64)) #Simulate from uniform to accept/reject

        if u < a #Accept proposal
            xout[n,:] .= xprime
            zout[n,:] .= zprime
            (x, z) = (xprime, zprime) #Update position in both Euclidean and Stereographic space
            
            aout += 1/(N-1) #Keep track of average acceptance probability
        else #Reject proposal
            xout[n,:] .= x
            zout[n,:] .= zprime
        end
    end
    println()

    return (x = xout, z = zout, a = aout)
end