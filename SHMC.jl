#Attempt at coding up a stereographic HMC

#Import Stereographic Projection stuff, and the Random library
include("Stereographic Projection.jl")
using Random

#RATTLE integrator for a potential u(z) and spherical constraint: take L RATTLE steps of length h
Rattle = function (gradu, z0, p0, h, L; gradz = missing)
    #Check that norm(z) == 1 and z.p=0
    abs(sum(z0.^2) -1) >= 1e-12 && error("norm(z) != 1")
    abs(sum(z0.*p0)) >= 1e-12 && error("z.p != 0")

    #Initialise position and momentum variables
    (z,p) = (z0,p0)

    #Initialise gradient and product
    ismissing(gradz) && (gradz = gradu(z))

    for i in 1:L
        #Initialise zstep vector (obtained by solving RATTLE equations)
        zstep = h*(p + h/2*(gradz - sum(gradz .* z)*z))
        zsteplength = sum(zstep.^2)
        
        #Check the step doesn't push us off the sphere
        if zsteplength >1
            #If we cannot solve the RATTLE equations, reject move and return to initial position
            return(z = z0, p = p0, gradz = gradz)
        end

        #Intermediate step storing the next position
        zprime = sqrt(1 - zsteplength)*z + zstep
        
        #Normalize z for mathematical stability
        normalize!(zprime)

        gradz = gradu(zprime) #Update gradient

        #Update velocity, then orthogonalize (again, found by solving RATTLE equations)
        p = h/2*gradz - 1/h*z
        p -= sum(p .* zprime)*zprime

        z = zprime #Fully update position
    end

    return(z = z, p = p)
end

#We simulate the Stereographic Hamiltonian MC path targeting the disribtuion f
SHMCSimulator = function (logf, x0, h, L, N; gradlogf = missing, sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)), includefirst = true)
    
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

    x = x0 #Position vector, initialised at x0

    #Calculate log-density on the sphere
    densz = logf(x) - d*log(1-z[end])

    aout = 0 #Track acceptance rate

    #Construct gradient of logf, if not specified
    if ismissing(gradlogf)
        d > 1 ? gradlogf = x -> ForwardDiff.gradient(logf,x) : gradlogf = x -> ForwardDiff.derivative(logf,x)
    end

    #Set up gradient of potential on the sphere
    gradu = SPgradlog(gradlogf)
    gradz = gradu(z; sigma=sigma, mu=mu).gradz
    
    for n in indexes
        #Print iteration number
        print("\rStep number: $n")

        #Initialise velocity, then orthogonalize
        p = randn(d+1)
        p -= sum(p .* z)*z

        #Take RATTLE step
        (zprime, pprime) = Rattle(z -> gradu(z; sigma=sigma, mu=mu).gradz, z, p, h, L; gradz = gradz)

        #Find projected point
        xprime = SP(zprime; sigma, mu)

        #Calculate log-density on the sphere
        densprime = logf(xprime) - d*log(1-zprime[end])

        #Compute log-acceptance probability, based on Hamiltonian of the dynamics
        a = 1/2*p'*p - densz - 1/2*pprime'*pprime + densprime

        u = log(rand(Float64)) #Simulate from uniform to accept/reject

        if u < a #Accept proposal
            #Update position in both Euclidean and Stereographic space
            (x,z) = (xprime,zprime)

            #Update log-density
            densz = densprime

            #Keep track of average acceptance probability
            aout += 1/(N - includefirst)
        end

        #Add to output
        xout[n,:] .= x
        zout[n,:] .= z
    end
    println()

    return (x = xout, z = zout, a = aout)
end