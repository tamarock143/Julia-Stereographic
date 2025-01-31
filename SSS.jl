### SSS Code ###

#Import Stereographic Projection stuff, and the Random library
include("Stereographic Projection.jl")
using Random

#We simulate a Stereographic Slice Sampler path targeting the disribtuion f
SliceSimulator = function(logf, x0, N; sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)), includefirst = true, steps = 1, printing = false)
    
    z = SPinv(x0; sigma = sigma, mu = mu, isinv = false) #Map to the sphere

    d = length(x0) #The dimension

    #Prepare output
    xout = zeros(N,d)
    zout = zeros(N,d+1)
    thetaout = zeros(N+includefirst)
    vout = zeros(N+includefirst,d+1)

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

    theta = undef #Predefining output angle

    Nprop = 0 #Predefine number of proposed points

    for n in indexes
        #Print iteration number
        printing && print("\rStep number: $n")

        #We only sample one point after several steps
        for _ in 1:steps
            t = log(rand()) + fx - d*log(1 - z[end]) #Sample the (log)height of the level set

            v = SBPSRefresh(z) #Sample velocity to determine which geodesic we are following
            vout[n-includefirst,:] .= v

            theta = 2pi*rand() #Sample initial angle around the geodesic

            Nprop += 1 #Increment number of proposals

            #Initialise bracketing interval for shrinkage
            thetamin = theta - 2pi
            thetamax = theta

            zprime = normalize(z*cos(theta) + v*sin(theta)) #New proposed point
            xprime = SP(zprime; sigma = sigma, mu = mu) #Project to Euclidean Space

            fxprime = logf(xprime) #Density at xprime

            #We additionally include a timer to ensure the algorithm does not run indefinitely
            k = 1

            #Follow the shrinkage procedure until we hit a point inside the super-level set
            while t >= fxprime - d*log(1 - zprime[end]) && k <= 1e6
                #On rejection, shrink the interval
                theta > 0 ? thetamax = theta : thetamin = theta

                #Resample position to be uniform inside the new interval
                theta = (thetamax - thetamin)rand() + thetamin
                
                Nprop += 1 #Increment number of proposals

                zprime = normalize(z*cos(theta) + v*sin(theta)) #New proposed point
                xprime = SP(zprime; sigma = sigma, mu = mu) #Project to Euclidean Space

                fxprime = logf(xprime) #Density at xprime

                k += 1 #Increment number of steps
            end

            #If we hit the step threshold, reject the output
            #Otherwise, update position in both Euclidean and Stereographic space
            k <= 1e6 && ((x, z, fx) = (xprime, zprime, fxprime))
        end

        #Add to output
        xout[n,:] .= x
        zout[n,:] .= z
        thetaout[n-includefirst] = theta
    end
    println()

    return (x = xout, z = zout, theta = thetaout, v = vout, Nprop = Nprop)
end