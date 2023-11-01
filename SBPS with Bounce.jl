#Import Stereographic Projection stuff, Brent stuff, and the Random library
include("Stereographic Projection.jl")
include("Brent's Method.jl")
using Random
using Plots
using SpecialFunctions
using StatsBase

#We simulate an SBPS path targeting the disribtuion f
#This requires bounce events and refreshment events
SBPSSimulator = function (f, x0, lambda, T, delta; logdens = false)
    #Start by ensuring we are working with a log-density
    !logdens && f = x -> log(f(x))

    z = SPinv(x0) #Map to the sphere
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

    while left > 0
        #Simulate next refreshment time according to Exp(lambda)
        tauref = randexp(Float64)/lambda

        t = min(tauref,left) #In case the path ends before the next event
        
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