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
    logdens || f = x -> log(f(x))

    #Precalculate the gradient function for f (or derivative in d=1 case)
    length(x0) > 1 ? gradf = x -> ForwardDiff.gradient(f,x) : gradf = x -> ForwardDiff.derivative(f,x)

    #This line is here to precalculated the gradient function, and create a global variable for the gradient vector
    mygrad = gradf(x0) 

    #Invert the matrix sigma
    sigmainv = inv(sigma)

    z = SPinv(x0; sigma = sigmainv, mu = mu, isinv = true) #Map to the sphere
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
    bouncerate = SBPSRate(gradf)

    while left > 0
        #Simulate next refreshment time according to Exp(lambda)
        tauref = randexp(Float64)/lambda

        #Simulate next event before time Tbrent
        #Time of potential bounce event
        taubounce = 0

        #Tracks how many chunks of Brent's method we have gone through
        l = 0
        #Indictor of whether we are still looking for a bounce
        nobounce = true

        while nobounce && taubounce < min(left,tauref)
            #Find upper bound M on the bounce rate for current chunk
            M = -Brent(s -> bouncerate(s,z,v; sigma = sigma, mu = mu)[1],
                     l*Tbrent, (l+1)Tbrent, tol)[2]

            #Is there a positive chance of a bounce?
            if M > 0
                #Simulate whether there will be a bounce in this chunk
                while nobounce
                    #Simulate next possible bounce time according to Exp(M)
                    taubounce += randexp(Float64)/M

                    #Did the event happen before the end of the interval?
                    taubounce > (l+1)Tbrent && (taubounce = (l+1)Tbrent; break)

                    #Did a bounce happen, or do we reject this event?
                    mybounce = bouncerate(taubounce,z,v; sigma, mu) #Rate and gradient at bounce time
                    u = rand(Float64) #Uniform random variable to accept/reject bounce

                    if -mybounce[1]/M > u #Accept bounce and break, or go for another loop
                        nobounce = false
                        mygrad = mybounce[2]
                    end
                end
            else
                #If negative rate, move to next interval
                taubounce = (l+1)Tbrent
            end

            l += 1 #Increment number of Brent steps so far
        end

        ########## Got to here
        
        #Time until next event, or need new upper bound
        t =
        
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