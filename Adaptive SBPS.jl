#Import SBPS Simulator
include("SBPS with Bounce.jl")

times = zeros(10)
beta = 0.5

for k in 1:10
    times[k] = 2^findfirst(map(x -> 2^x, 0:10) .>= k^beta)
end

times

SBPSSimulator = function (gradlogf, x0, lambda, T, delta, beta, r, R; Tbrent = pi/24, tol = 1e-6,
    sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)))

    z = SPinv(x0; sigma = sigma, mu = mu, isinv = false) #Map to the sphere
    v = SBPSRefresh(z) #Initialize velocity
    d = length(x0) #The dimension

    n = floor(BigInt, T/delta)+1 #Total number of observations of the skeleton path

    #Prepare output
    zout = zeros(n,d+1)
    vout = zeros(n,d+1)
    xout = zeros(n,d)
    
    totalleft::Float64 = (n-1)delta #Track remaining amount of time left until last observation
    k = 2 #Track the next row to be added to output
    t0::Float64 = delta #Amount of time after an event until the next skeleton path sample time

    #Placeholder for gradient variable
    mygrad = zeros(d+1)

    #Set up Bounce rate function
    bouncerate = SBPSRate(gradlogf)

    #Set up adaptation times
    nadapt = ceil(((beta+1)T)^(1/(beta+1)))
    times = zeros(nadapt)

    for i in 1:nadapt
        times[i] = 2^findfirst(map(x -> 2^x, 0:10) .>= i^beta)
    end

    #Prepare estimators for mu and sigma
    #m and s2 track sums we will need to iteratively update the estimators
    m = zeros(d)
    s2 = I(d)

    #muest and sigmaest are the estimators
    muest = zeros(nadapt+1, d)
    sigmaest = fill(sqrt(d)I(d),nadapt+1)

    muest[1,:] = mu
    sigmaest[1] = sigma
    
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
        
        #Perform Gram-Schmidt on (z,v) to account for incremental numerical errors
        normalize!(z)
        v = normalize(v - sum(z.*v)z)
    end

    #Ensure we have added final position and velocity
    zout[n,:] = z
    vout[n,:] = v
end