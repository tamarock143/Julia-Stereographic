#Import SRW Simulator and Optimisation library
include("SRW.jl")
include("Optimisation.jl")

SRWAdaptive = function(logf, x0, h2, N, beta, r, R;
    sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)), burnin = 1e2, adaptlength = burnin)

    d = length(x0) #The dimension

    d == 1 && (x0 = fill(x0, 1))

    #Prepare output
    zout = zeros(N,d+1)
    xout = zeros(N,d)

    xout[1,:] = x0
    
    left::Int64 = N #Track remaining amount of steps left until last observation
    
    #Ensure burn-in is not close to 0 for ease of future calculations
    burnin <= 1 && (burnin = 1)

    #Set up adaptation times
    #Include initial burn-in period
    times::Vector{Int64} = [burnin]
    i=1

    #Variable tracking which power of 2 is used in length of the adaptative epoch
    power = 0

    #Tracker for whether we have enough adaptations to fill the time
    timescheck = left - times[1]

    while timescheck > 0
        #For theoretical guarantees, we ensure increasing common divisor to lags
        
        #Increment powers of 2 to be of order i^beta
        power += findfirst(map(x -> 2^x, power:power+beta+1) .>= i^beta) -1

        #Add on the next adaptive epoch
        append!(times, min(adaptlength*2^power, timescheck))

        #Remove this epoch's length
        timescheck -= times[end]

        #Iterate length of times
        i += 1
    end

    #Exact number of adaptive times
    nadapt = i

    #Indexes of starts of each adaptation in xout. Will be required for moving paths into output
    adaptstarts = append!([1], cumsum(times)[1:end] .+ 1)

    #Prepare estimators for mu and sigma
    #m and s2 track sums we will need to iteratively update the estimators
    m::Vector{Vector{Float64}} = fill(zeros(d), nadapt)
    s2::Vector{Matrix{Float64}} = fill(zeros(d,d), nadapt)

    #muest and sigmaest are the estimators
    muest = zeros(nadapt+1, d)
    sigmaest = fill(zeros(d,d),nadapt+1)

    muest[1,:] = mu
    sigmaest[1] = sigma

    iadapt = 1 #Track which adaptive estimate we are currently using

    #Return Robbins-Monro scaling constants
    cout = zeros(nadapt)

    #For each adaptive period, run the SBPS simulation with fixed parameter values
    for t in times
        #If we've run the full time, end the process
        left == 0 && break

        println("Adaptation number: ", iadapt, "/", nadapt, ". Adaptation length: ", t, "\n")
        
        #Run the process with the given parameters
        @time (xpath,zpath) = SRWSimulator(logf, xout[adaptstarts[iadapt],:], h2, min(t,left); 
        sigma = sigmaest[iadapt], mu = muest[iadapt,:], includefirst = (iadapt == 1))

        #Update how much time is left
        left >= t ? left -= t : left = 0

        #Input changes based on whether we need to include the first value
        if iadapt == 1
            #Append this piece to the output
            zout[adaptstarts[iadapt]:adaptstarts[iadapt+1]-1,:] = zpath
            xout[adaptstarts[iadapt]+1:adaptstarts[iadapt+1]-1,:] = xpath[2:end,:]
        else
            #Append this piece to the output
            zout[adaptstarts[iadapt]:adaptstarts[iadapt+1]-1,:] = zpath
            xout[adaptstarts[iadapt]:adaptstarts[iadapt+1]-1,:] = xpath
        end

        #Update column sums
        m[iadapt] = vec(sum(xpath, dims = 1))

        #We will "forget" the first half of the epochs
        #Calculate how many terms are used for the estimators
        nlearn = sum(times[ceil(Int64, iadapt/3):iadapt])

        #Placeholder for mu update
        mutemp = sum(m[ceil(Int64, iadapt/3):iadapt])/nlearn

        #Update sum for covariance estimator
        for x in eachrow(xpath)
            s2[iadapt] += x*x'
        end

        #Reduce norm of mean estimate if necessary
        rtemp = sum(mutemp.^2)
        rtemp > R^2 && (mutemp *= R/rtemp)

        #Output mean estimator
        muest[iadapt+1,:] = mutemp

        #Try statement included in case of NaNs or other matrix irregularities (such as near-0 eigenvalues)
        try
            #Placeholder for sigma update
            sigmatemp = nlearn/(nlearn-1)*Symmetric(sum(s2[ceil(Int64, iadapt/3):iadapt])/nlearn - mutemp*mutemp')

            invtemp = inv(sqrt(sigmatemp))

            #Create set of norms for centered and scaled output
            #Using many epochs to tune the shape, we use only the latest data for the full scale
            xnorms = Vector{Float64}()
            for x in eachrow(xpath[floor(Int64,size(xpath)[1]/2):end,:])
                append!(xnorms, sum(x -> x^2, invtemp*(x - mutemp)))
            end

            #We now scale the covariance matrix so that the latitude is centered
            #If the estimators are correct, should get c=d
            #For this, we use the Robbins-Monro algorithm (note this requires an increasing functional)

            #latf(theta) = mean(x -> (x - theta)/(x + theta), xnorms)
            #latgrad(theta) = mean(x -> -2x/(x - theta)^2, xnorms)
            #c = Newton(latf, latgrad, d, tol)

            latf(x,theta) = -(x - theta)/(x + theta) #Negative Latitude of a given z at position ||x||^2/theta
            c = RobMonro(latf, xnorms, d, 1, 1e6; lower = (10d)^-10, upper = (10Float64(d))^10)
            println(sqrt(c/d))
            
            #Scale the covariance
            sigmatemp *= c
            cout[iadapt] = c

            #Diagonalise the covariance estimator
            (evalstemp,evecstemp) = eigen(sigmatemp)

            #Truncate large eigenvalues
            for i in 1:d
                evalstemp[i] > R^2 && (evalstemp[i] = R^2)
                evalstemp[i] < 0 && (evalstemp[i] = 0)
            end

            #Output sigma estimator, equal to sqrt of covariance estimator
            #We include the +r*I(d) term to get lower bounds on the eigenvalues
            sigmaest[iadapt+1] = Symmetric(evecstemp*Diagonal(sqrt.(evalstemp))*evecstemp') + r*I(d)
        catch
            #If there was an error, just keep the previous estimate
            sigmaest[iadapt+1] = sigmaest[iadapt]
            println("Warning: Covariance Matrix Update Error")
        end
        
        #Increment number of adaptations
        iadapt += 1
    end

    return (x = xout, z = zout, mu = muest, sigma = sigmaest, times = times, c = cout)
end