#Import Stereographic Projection stuff, Brent stuff, and the Random library
include("Stereographic Projection.jl")
include("Optimisation.jl")
using Random

#We simulate an SBPS path targeting the disribtuion f. It requires x -> ∇log(f(x))
#This requires bounce events and refreshment events
SBPSSimulator = function(gradlogf, x0, lambda, T, delta; w = missing, Tbrent = 1, Epsbrent = 0.01, tol = 1e-6,
        sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)), printing = false)
    
    z = SPinv(x0; sigma = sigma, mu = mu, isinv = false) #Map to the sphere

    d = length(x0) #The dimension

    #Check that w is a unit vector (or missing)
    ismissing(w) || abs(sum(w.^2) -1) <= 1e-12 || error("norm(w) != 1")

    #Initialize velocity. If we have a spcified w, use it for the initial velocity
    if ismissing(w)
        v = SBPSRefresh(z)
    else
        v = zeros(d+1)
        v[d+1] = sum(w.*z[1:d])
        v[1:d] = w[1:d] - v[d]*z[1:d]/(1-z[d+1])
        v = normalize(v - sum(z.*v)z) #Normalize for regularity
    end

    n = floor(BigInt, T/delta)+1 #Total number of observations of the skeleton path

    #Prepare output
    zout = zeros(n,d+1)
    vout = zeros(n,d+1)

    zout[1,:] = z
    vout[1,:] = v

    left::Float64 = (n-1)delta #Track remaining amount of time left until last observation
    k = 2 #Track the next row to be added to output
    t0::Float64 = delta #Amount of time after an event until the next skeleton path sample time

    #Placeholder for gradient variable
    mygrad = zeros(d+1)

    #Set up Bounce rate function
    bouncerate = SBPSRate(gradlogf)

    bounceindic = Vector{Bool}()

    #Start counter of number of gradient evaluations, separated between those used in optimisation and thinning
    Nopt = Nthin = 0

    while left > 0
        #Simulate next refreshment time according to Exp(lambda)
        tauref = randexp(Float64)/lambda

        #Simulate next event before time Tbrent
        #Time of potential bounce event
        taubounce = 0

        #Tracks how many chunks of Brent's method we have gone through
        Bmin = 0
        Bmax = Tbrent*(1 + Epsbrent - z[end]^2)
        #Indictor of whether we are still looking for a bounce
        nobounce = true

        while nobounce && taubounce <= min(left,tauref)
            #Find upper bound M on the bounce rate for current chunks
            #We flip the sign of the bound because we are minimising -lambda(z,v)
            (M,tempevals) = (-1,1) .* Brent(s -> bouncerate(s,z,v; sigma = sigma, mu = mu)[1],
                    Bmin, Bmax, tol; countevals = true)[2:3]
            
            Nopt += tempevals #Brent's method outputs number of gradient evaluations

            #Is there a positive chance of a bounce?
            if M > 0
                #Simulate whether there will be a bounce in this chunk
                while nobounce
                    #Simulate next possible bounce time according to Exp(M)
                    taubounce += randexp(Float64)/M

                    #Did the event happen before the end of the interval or another event?
                    taubounce > min(Bmax,tauref,left) && (taubounce = Bmax; break)

                    #Did a bounce happen, or do we reject this event?
                    mybounce = bouncerate(taubounce,z,v; sigma, mu) #Rate and gradient at bounce time
                    Nthin += 1 #Bouncerate requires 1 gradient evaluation

                    u = rand(Float64) #Uniform random variable to accept/reject bounce

                    if -mybounce[1]/M > u #Accept bounce and break, or go for another loop
                        nobounce = false
                        mygrad = mybounce[2]
                    end
                end
            else
                #If negative rate, move to next interval
                taubounce = Bmax
            end

            #Update Brent interval
            Bmin = Bmax
            Bmax += Tbrent*(1 + Epsbrent - (z[end]*sin(Bmax) + v[end]*cos(Bmax))^2)
        end

        push!(bounceindic, !nobounce)

        #Time until next event, or need new upper bound
        t = min(left, tauref, taubounce)

        #If the next observation time is after the event time, we do not need to simulate the path
        if t0 > t
            #Update next observation time
            t0 -= t
        else
            #Simulate the next piece of the path
            (zpath,vpath) = SBPSDetPath(z,v,t,delta; t0)

            #Number of rows to add
            nadd = size(zpath)[1]

            #Append this piece to the output
            zout[k:k+nadd-1,:] = zpath
            vout[k:k+nadd-1,:] = vpath

            #Increment next row to add
            k += nadd

            #Time to next observation time
            t0 += (floor((t-t0)/delta)+1)delta -t
        end

        #Update position and velocity based on whether we had a refreshment event
        if taubounce < min(tauref,left)
            #Update position and temporarily set update velocity as required to bounce
            (z,v) = (cos(t)z + sin(t)v, cos(t)v - sin(t)z)

            #For bounce event, update velocity using SBPSBounce and gradient
            v = SBPSBounce(z,v,mygrad)
        elseif tauref < left
            z = cos(t)z + sin(t)v
            #For refreshment event, update velocity using SBPSRefresh
            v = SBPSRefresh(z)
        else
            #If no event occured before the end of the path, simply follow the trajectory
            (z,v) = (cos(t)z + sin(t)v, cos(t)v - sin(t)z)
        end

        #Update remaining path length
        left -= t 
        
        #Print time left
        printing && print("Time left: $left\r")
        
        #Perform Gram-Schmidt on (z,v) to account for incremental numerical errors
        normalize!(z)
        v = normalize(v - sum(z.*v)z)
    end

    #Ensure we have added final position and velocity
    zout[n,:] = z
    vout[n,:] = v

    #Remove final event, since it records the "end" event
    pop!(bounceindic)

    return (z = zout, v = vout, events = bounceindic, Nevals = [Nopt, Nthin])
end

#SBPS simulator with geometrically adapting Brent's method window
SBPSGeom = function(gradlogf, x0, lambda, T, delta; w = missing, Tbrent = 1, Abrent = 1.1, Nbrent = 1, tol = 1e-6,
    sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)), printing = false)

    z = SPinv(x0; sigma = sigma, mu = mu, isinv = false) #Map to the sphere

    d = length(x0) #The dimension

    #Check that w is a unit vector (or missing)
    ismissing(w) || abs(sum(w.^2) -1) <= 1e-12 || error("norm(w) != 1")

    #Initialize velocity. If we have a spcified w, use it for the initial velocity
    if ismissing(w)
        v = SBPSRefresh(z)
    else
        v = zeros(d+1)
        v[d+1] = sum(w.*z[1:d])
        v[1:d] = w[1:d] - v[d]*z[1:d]/(1-z[d+1])
        v = normalize(v - sum(z.*v)z) #Normalize for regularity
    end

    n = floor(BigInt, T/delta)+1 #Total number of observations of the skeleton path

    #Prepare output
    zout = zeros(n,d+1)
    vout = zeros(n,d+1)

    zout[1,:] = z
    vout[1,:] = v

    left::Float64 = (n-1)delta #Track remaining amount of time left until last observation
    k = 2 #Track the next row to be added to output
    t0::Float64 = delta #Amount of time after an event until the next skeleton path sample time

    #Placeholder for gradient variable
    mygrad = zeros(d+1)

    #Set up Bounce rate function
    bouncerate = SBPSRate(gradlogf)

    bounceindic = Vector{Bool}()

    #Start counter of number of gradient evaluations, separated between those used in optimisation and thinning
    Nopt = Nthin = 0

    while left > 0
        #Simulate next refreshment time according to Exp(lambda)
        tauref = randexp(Float64)/lambda

        #Simulate next event before time Tbrent
        #Time of potential bounce event
        taubounce = 0
        #Time between latest events
        taustep = 0

        #Tracks the starting point for the thinning upper bound
        Bmin = 0
        Bmax = Bmin + Tbrent
        #Indictor of whether we are still looking for a bounce
        nobounce = true

        while nobounce && taubounce < min(left,tauref)
            #For step function upper bound, divide into Nbrent windows of equal length
            twindows = repeat([Tbrent/Nbrent], Nbrent)
            M = zeros(Nbrent)

            for i in 1:Nbrent
                #Find upper bound M on the bounce rate for current chunks
                #We flip the sign of the bound because we are minimising -lambda(z,v)
                (M[i],tempevals) = (-1,1) .* Brent(s -> bouncerate(s,z,v; sigma = sigma, mu = mu)[1],
                        Bmin + (i-1)*Tbrent/Nbrent, Bmin + i*Tbrent/Nbrent, tol; countevals = true)[2:3]
                
                Nopt += tempevals #Brent's method outputs number of gradient evaluations
            end

            #Set the rate upper bounds to be positive
            map!(m -> max(0,m), M, M)

            #Simulate whether there will be a bounce in this chunk
            while nobounce && taubounce < min(left,tauref,Bmax)
                #Simulate next possible bounce time according to the Poisson Process with the upper bound rate
                (taustep, boundtemp) = PoissonStepSim(M,twindows; tau0 = taustep)
                #Update time of final potential bounce event
                taubounce += taustep

                #Did the event happen before the end of the interval or another event?
                if isinf(taubounce)
                    #Reset ourselves at the start of the next window
                    taubounce = Bmax
                    taustep = 0
                    
                    #If we had no event in the window, increase future window width
                    Tbrent *= Abrent
                    #Since the paths are periodic, only need to search up to 2pi
                    Tbrent > 2pi && (Tbrent = 2pi)

                elseif taubounce < min(left,tauref) #If we do bounce, but another event happened first, skip

                    #Thinning step: did a bounce happen, or do we reject this event?
                    mybounce = bouncerate(taubounce,z,v; sigma, mu) #Rate and gradient at bounce time
                    Nthin += 1 #Bouncerate requires 1 gradient evaluation

                    u = rand(Float64) #Uniform random variable to accept/reject bounce

                    #Sanity check for the upper bound
                    -mybounce[1] > boundtemp && error("Invalid upper bound")

                    if -mybounce[1]/boundtemp > u #Accept bounce and break, or go for another loop
                        nobounce = false
                        mygrad = mybounce[2]
                    else
                        #If we rejected the proposal, shrink future window width
                        Tbrent /= Abrent
                        #If each window would be below the tolerance width, we would waaste computing time
                        Tbrent < tol*Nbrent && (Tbrent = tol*Nbrent)
                    end
                end
            end

            #Update Brent starting point
            Bmin = Bmax
            Bmax += Tbrent
        end

        push!(bounceindic, !nobounce)

        #Time until next event, or need new upper bound
        t = min(left, tauref, taubounce)

        #If the next observation time is after the event time, we do not need to simulate the path
        if t0 > t
            #Update next observation time
            t0 -= t
        else
            #Simulate the next piece of the path
            (zpath,vpath) = SBPSDetPath(z,v,t,delta; t0)

            #Number of rows to add
            nadd = size(zpath)[1]

            #Append this piece to the output
            zout[k:k+nadd-1,:] = zpath
            vout[k:k+nadd-1,:] = vpath

            #Increment next row to add
            k += nadd

            #Time to next observation time
            t0 += (floor((t-t0)/delta)+1)delta -t
        end

        #Update position and velocity based on whether we had a refreshment event
        if taubounce < min(tauref,left)
            #Update position and temporarily set update velocity as required to bounce
            (z,v) = (cos(t)z + sin(t)v, cos(t)v - sin(t)z)

            #For bounce event, update velocity using SBPSBounce and gradient
            v = SBPSBounce(z,v,mygrad)
        elseif tauref < left
            z = cos(t)z + sin(t)v
            #For refreshment event, update velocity using SBPSRefresh
            v = SBPSRefresh(z)
        else
            #If no event occured before the end of the path, simply follow the trajectory
            (z,v) = (cos(t)z + sin(t)v, cos(t)v - sin(t)z)
        end

        #Update remaining path length
        left -= t 
        
        #Print time left
        printing && print("Time left: $left\r")
        
        #Perform Gram-Schmidt on (z,v) to account for incremental numerical errors
        normalize!(z)
        v = normalize(v - sum(z.*v)z)
    end

    #Ensure we have added final position and velocity
    zout[n,:] = z
    vout[n,:] = v

    #Remove final event, since it records the "end" event
    pop!(bounceindic)

    return (z = zout, v = vout, events = bounceindic, Nevals = [Nopt, Nthin], Tbrent = Tbrent)
end


#SBPS Simulator outputing only event skeleton points
SBPSEventSimulator = function (gradlogf, x0, lambda, T; Tbrent = pi/24, tol = 1e-6,
    sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)))

    z = SPinv(x0; sigma = sigma, mu = mu, isinv = false) #Map to the sphere
    v = SBPSRefresh(z) #Initialize velocity
    d = length(x0) #The dimension

    #Initialize output
    zout = [z]
    vout = [v]

    eventsout = ["Start"]

    left::Float64 = T #Track remaining amount of time left until last observations

    #Placeholder for gradient variable
    mygrad = zeros(d+1)

    #Set up Bounce rate function
    bouncerate = SBPSRate(gradlogf)

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

        while nobounce && taubounce <= min(left,tauref)
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

        #Time until next event, or need new upper bound
        t = min(left, tauref, taubounce)

        #Update position and velocity based on whether we had a refreshment event
        if taubounce < min(tauref,left)
            #Update position and temporarily set update velocity as required to bounce
            (z,v) = (cos(t)z + sin(t)v, cos(t)v - sin(t)z)

            #For bounce event, update velocity using SBPSBounce and gradient
            v = SBPSBounce(z,v,mygrad)

            #Indicate it is a Bounce event
            push!(eventsout, "Bounce")
        elseif tauref < left
            z = cos(t)z + sin(t)v
            #For refreshment event, update velocity using SBPSRefresh
            v = SBPSRefresh(z)

            push!(eventsout, "Refresh")
        else
            #If no event occured before the end of the path, simply follow the trajectory
            (z,v) = (cos(t)z + sin(t)v, cos(t)v - sin(t)z)

            push!(eventsout, "End")
        end
    
        #Perform Gram-Schmidt on (z,v) to account for incremental numerical errors
        normalize!(z)
        v = normalize(v - sum(z.*v)z)

        push!(zout,z)
        push!(vout,v)
        
        #Update remaining path length
        left -= t
    end

    return (z = zout, v = vout, events = eventsout)
end