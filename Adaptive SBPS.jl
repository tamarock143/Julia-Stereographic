#Import SBPS Simulator
include("SBPS with Bounce.jl")

times = zeros(10)
beta = 0.5

for k in 1:10
    times[k] = 2^findfirst(map(x -> 2^x, 0:10) .>= k^beta)
end

times

SBPSAdaptive = function(gradlogf, x0, lambda, T, delta, beta, r, R; Tbrent = pi/24, tol = 1e-6,
    sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)))

    d = length(x0) #The dimension

    n = floor(BigInt, T/delta)+1 #Total number of observations of the skeleton path

    #Prepare output
    zout = zeros(n,d+1)
    vout = zeros(n,d+1)
    xout = zeros(n,d)

    xout[1,:] = x0
    
    left::Float64 = (n-1)delta #Track remaining amount of time left until last observation
    k = 2 #Track the next row to be added to output
    
    #Set up adaptation times
    nadapt = ceil(((beta+1)left)^(1/(beta+1)))
    times = zeros(nadapt)

    for i in 1:nadapt
        #For theoretical guarantees, we ensure increasing common divisor to lags
        #For practical reasons, ensure adaptation times are an integer number of skeleton steps
        times[i] = ceil(1/delta)delta*2^findfirst(map(x -> 2^x, 0:10) .>= i^beta)
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

    iadapt = 1 #Track which adaptive estimate we are currently using

    #For each adaptive period, run the SBPS simulation with fixed parameter values
    for t in times
        #Run the process with the given parameters
        (zpath,vpath) = SBPSSimulator(gradlogf, xout[k-1], lambda, min(t,left), delta; 
        Tbrent, tol, sigma = sigmaest[iadapt], mu = muest[iadapt,:])

        left >= t ? left -= t : left = 0
    end
    
    #Ensure we have added final position and velocity
    zout[n,:] = z
    vout[n,:] = v
end