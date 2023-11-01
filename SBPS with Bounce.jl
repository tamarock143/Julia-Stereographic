#Import Stereographic Projection stuff, Brent stuff, and the Random library
include("Stereographic Projection.jl")
include("Brent's Method.jl")
using Random
using Plots
using SpecialFunctions
using StatsBase

#We simulate an SBPS path targeting the disribtuion f
#This requires bounce events and refreshment events
SBPSSimulator = function (f, x0, lambda, T, delta; Tbrent = pi/4, tol = 1e-9,
        sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)),
        logdens = false)
    
    #Start by ensuring we are working with a log-density
    !logdens && f = x -> log(f(x))

    #Precalculate the gradient function for f
    gradf = x -> ForwardDiff.gradient(f,x)

    #This line is here to precalculated the gradient function, and create a global variable for the gradient vector
    mygrad = gradf(x0) 

    #Invert the matrix sigma
    sigmainv = inv(sigma)

    z = SPinv(x0; sigmainv, mu, isinv = true) #Map to the sphere
    v = SBPSRefresh(z) #Initialize velocity
    d = length(x0) #The dimension

    n = floor(BigInt, T/delta)+1 #Total number of observations of the skeleton path

    #Prepare output
    zout = zeros(n,d+1)
    vout = zeros(n,d+1)

    zout[1,:] = z
    vout[1,:] = v

    left::Float64 = (n-1)delta #Track remaining amount of time left until last observation
    k = 2 #Track the next row to be added to output
    t0::Float64 = delta #Amount of time after an event until the next skeleton path sample time

    #Set up Bounce rate function
    bouncerate = SBPSBounce(gradf)

    while left > 0
        #Simulate next event before time Tbrent
        #Tbound will track how much time the Brent bound on the bounce rate still holds for
        Tbound = min(left, Tbrent)

        #Find upper bound M on the bounce rate
        M = -Brent(s -> bouncerate(s,z,v; sigma, mu)[1], 0, Tbound, tol)

        #Time of potential bounce event
        taubounce = 0
        #Indictor of whether we are still looking for a bounce
        bounce = false

        #Simulate the bounce event time up to Tbound
        while !bounce
            #Simulate next possible bounce time according to Exp(M)
            taubounce += randexp(Float64)/M

            #Did the event happen before the end of the interval?
            taubounce > Tbound && (taubounce = Tbound; break)

            #Did a bounce happen, or do we need a new bound?
            mybounce = bouncerate(taubounce,z,v; sigma, mu) #Rate and gradient at bounce time
            u = rand(Float64) #Uniform random variable to accept/reject bounce

            if mybounce[1]/M > u #Accept bounce and break, or go for another loop
                bounce = true
                mygrad = mybounce[2]
            end
        end

        ########## Got to here
        
        #Update remaining path length
        left -= t 

        #If the next observation time is after the event time, we do not need to simulate the path
        if t0 > t
            #Update next observation time
            t0 -= t
        else
            #Simulate the next piece of the path
            (zpath,vpath) = SBPSDetPath(z,v,t,delta; t0)

            #Number of rows to add.
            nadd = size(zpath)[1]

            k + nadd -1 > n && error("something strange happened with adding rows")

            #Append this piece to the output
            zout[k:k+nadd-1,:] = zpath
            vout[k:k+nadd-1,:] = vpath

            #Increment next row to add
            k += nadd

            #Time to next observation time
            t0 += (floor((t-t0)/delta)+1)delta -t
        end
        
        #Update position and velocity based on whether we had a refreshment event
        if tauref <= left
            z = cos(t)z + sin(t)v
            v = SBPSRefresh(z)
        else
            (z,v) = (cos(t)z + sin(t)v, cos(t)v - sin(t)z)
        end
    end

    #Ensure we have added final position and velocity
    zout[n,:] = z
    vout[n,:] = v

    return (z = zout, v = vout)
end