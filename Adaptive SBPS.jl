#Import SBPS Simulator
include("SBPS with Bounce.jl")

SBPSAdaptive = function(gradlogf, x0, lambda, T, delta, beta, r, R; Tbrent = pi/24, tol = 1e-6,
    sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)), burnin = 1e2)

    d = length(x0) #The dimension

    n = floor(BigInt, T/delta)+1 #Total number of observations of the skeleton path

    #Prepare output
    zout = zeros(n,d+1)
    vout = zeros(n,d+1)
    xout = zeros(n,d)

    xout[1,:] = x0
    
    left::Float64 = (n-1)delta #Track remaining amount of time left until last observation
    k = 2 #Track the next row to be added to output
    
    #Ensure burn-in is not close to 0 for ease of future calculations
    burnin <= 1 && (burnin = 1)

    #Set up adaptation times
    #Note this is not a tight upper bound on number of adaptations
    nadapt = ceil(Int64, ((beta+1)*(left-burnin)/burnin)^(1/(beta+1)))+1
    times = zeros(nadapt)

    #Include initial burn-in period
    times[1] = burnin

    i=2

    while sum(times) < left
        #For theoretical guarantees, we ensure increasing common divisor to lags
        #For practical reasons, ensure adaptation times are an integer number of skeleton steps
        times[i] = burnin*ceil(1/delta)delta*2^findfirst(map(x -> 2^x, 0:nadapt) .>= i^beta)

        #Iterate length of times
        i += 1
    end

    #Shrink length of adaptive times vector
    nadapt = i-1
    times = times[1:nadapt]

    #Prepare estimators for mu and sigma
    #m and s2 track sums we will need to iteratively update the estimators
    m::Vector{Float64} = mu
    s2::Matrix{Float64} = sigma

    #muest and sigmaest are the estimators
    muest = zeros(nadapt+1, d)
    sigmaest = fill(zeros(d,d),nadapt+1)

    muest[1,:] = mu
    sigmaest[1] = sigma

    iadapt = 1 #Track which adaptive estimate we are currently using

    #For each adaptive period, run the SBPS simulation with fixed parameter values
    for t in times
        #If we've run the full time, end the process
        left == 0 && break

        println("Adaptations so far: ", iadapt, "/", nadapt)
        
        #Run the process with the given parameters
        (zpath,vpath) = SBPSSimulator(gradlogf, xout[k-1,:], lambda, min(t,left), delta; 
        Tbrent, tol, sigma = sigmaest[iadapt], mu = muest[iadapt,:])

        #Update how much time is left
        left >= t ? left -= t : left = 0
        
        #Number of rows to add (note that the start of the current path is the end of the previous one for z)
        pathlength = size(zpath)[1]

        #Need small change based on whether we have put in the first entries for (zout,vout)
        if k >2
            #Append this piece to the output
            zout[k:k+pathlength-2,:] = zpath[2:end,:]

            #Note we will have difficulty with storing the velocities at adaptation times
            vout[k-1:k+pathlength-3,:] = vpath[1:end-1,:]

            #Input final velocity value if needed
            left == 0 && (vout[end,:] = vpath[end,:])
        else
            #Append this piece to the output
            zout[1:pathlength,:] = zpath

            #Note we will have difficulty with storing the velocities at adaptation times
            vout[1:pathlength-1,:] = vpath[1:end-1,:]
        end

        #Set up x projected path
        xpath = zeros(pathlength-1,d)

        #Project each entry back to R^d
        for i in 2:pathlength
            xpath[i-1,:] = SP(zpath[i,:]; sigma = sigmaest[iadapt], mu = muest[iadapt])
        end

        #Record this section of the x path
        xout[k:k+pathlength-2,:] = xpath

        #Increment next row to add
        k += pathlength -1

        #Update column sums
        m .+= sum(xpath, dims = 1)'

        #Placeholder for mu update
        mutemp = m/k

        #Update sum for covariance estimator
        for i in 1:k-1
            s2 += (xout[i,:] - mutemp)*(xout[i,:] - mutemp)'
        end

        #Placeholder for sigma update
        sigmatemp = Symmetric(d*s2/(k-1))

        #Increment which adaptive step we are at
        iadapt += 1

        #Reduce norm of mean estimate if necessary
        (rtemp = sqrt(sum(mutemp.^2))) > R ? muest[iadapt,:] = R*mutemp/rtemp : muest[iadapt,:] = mutemp

        #Diagonalise the covariance estimator
        (evalstemp,evecstemp) = eigen(sigmatemp)

        #Truncate eigenvalues
        for i in 1:d
            if evalstemp[i] > R^2
                evalstemp[i] = R^2
            elseif evalstemp[i] < r^2
                evalstemp[i] = r^2
            end
        end
        
        #Output sigma estimator, equal to sqrt of covariance estimator
        sigmaest[iadapt] = evecstemp*Diagonal(sqrt.(evalstemp))*evecstemp'
    end

    return (x = xout, z = zout, v = vout, mu = muest, sigma = sigmaest, times = times)
end