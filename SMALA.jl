

SMALA = function(logf, x0, h, N; gradlogf = missing, sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)), includefirst = true, steps = 1, printing = false)
    
    z = SPinv(x0; sigma = sigma, mu = mu, isinv = false) #Map to the sphere

    d = length(x0) #The dimension

    #Prepare output
    xout = zeros(N,d)
    zout = zeros(N,d+1)

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

    #Construct gradient of logf, if not specified
    if ismissing(gradlogf)
        d > 1 ? gradlogf = x -> ForwardDiff.gradient(logf,x) : gradlogf = x -> ForwardDiff.derivative(logf,x)
    end

    x = x0 #Position vector, initialised at x0

    fx = logf(x) #Precalculate density at position

    #Set up gradient of potential on the sphere
    gradstereo = SPgradlog(gradlogf)
    gradz = gradstereo(z; sigma=sigma, mu=mu).gradz

    #Orthogonalise gradient
    gradz -= sum(z .* gradz)*z

    aout = 0 # Track acceptance rate
    
    for n in indexes
        #Print iteration number
        printing && print("\rStep number: $n")

        #We only sample one point after several steps
        for _ in 1:steps
            dz = h*randn(d+1) #Gaussian step
            dz -= sum(z.*dz)*z #Project step onto the tangent plane at z

            zprime = normalize(z + h^2/2*gradz + dz) #New proposed point

            (gradprime, xprime) = gradstereo(zprime; sigma=sigma, mu=mu)
            
            #Orthogonalise gradient
            gradprime -= sum(zprime .* gradprime)*zprime

            fxprime = logf(xprime) #Density at xprime

            zdot = sum(z .* zprime)

            #Compute log-acceptance probability, based on projected density
            a = (-fx + d*log(1 - z[end]) + fxprime - d*log(1 - zprime[end]) 
                - 1/(2h^2)*norm(z/zdot - zprime - h^2/2* gradprime)^2 
                + 1/(2h^2)*norm(zprime/zdot - z - h^2/2* gradz)^2) #Note that dz = 

            u = log(rand(Float64)) #Simulate from uniform to accept/reject

            if u < a #Accept proposal
                #Update position in both Euclidean and Stereographic space
                (x, z, fx, gradz) = (xprime, zprime, fxprime, gradprime) 

                #Keep track of number of accepts
                aout += 1
            end
        end
        
        #Add to output
        xout[n,:] .= x
        zout[n,:] .= z
    end
    println()

    return (x = xout, z = zout, a = aout/(N*steps - includefirst))
end