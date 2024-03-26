#Attempt at coding up a stereographic HMC

#Import Stereographic Projection stuff, and the Random library
include("Stereographic Projection.jl")
using Random

#RATTLE integrator for Spherical constraint: take L RATTLE steps of length h
Rattle = function (gradlogf, z0, p0, h, L; gradz = missing)
    #Check that norm(z) == 1 and z.p=0
    abs(sum(z0.^2) -1) >= 1e-12 && error("norm(z) != 1")
    abs(sum(z0.*p0)) >= 1e-12 && error("z.p != 0")

    #Initialise position and momentum variables
    (z,p) = (z0,p0)

    #Initialise gradient and product
    ismissing(gradz) && (gradz = gradlogf(z))
    graddotz = sum(gradz.*z)

    for i in 1:L
        #Check the discriminant to ensure lambda exists
        if (disc = graddotz^2 + 4/h^4 - 4/h^2*sum(x -> x^2, p - h/2*gradz)) < 0
            #If we cannot solve the RATTLE equations, reject move and return to initial position
            return(z = z0, p = p0)
        end
        #Initialise the Lagrange multiplier lambda
        lambda = 2/h^2 + graddotz - sqrt(disc)
        
        p += h/2*(gradz - lambda*z) #First "half-update" for p

        z += h*p #Update z

        #Normalize z for mathematical stability
        normalize!(z)

        #Update gradient and product
        gradz = gradlogf(z)
        graddotz = sum(gradz.*z)

        #Second "half-update" for p
        p = vec((I - z*z')*(p + h/2*gradz))
    end

    return(z = z, p = p, gradz = gradz)
end

#We simulate the Stereographic Hamiltonian MC path targeting the disribtuion f
SHMCSimulator = function (gradlogf, x0, h, L, N; sigma = sqrt(length(x0))I(length(x0)), mu = zeros(length(x0)), includefirst = true)
    
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

    aout = 0 #Track acceptance rate

    #Set up gradient on the sphere
    gradlogf = SPgradlog(gradlogx)
    gradz = gradlogf(z; sigma=sigma, mu=mu)
    
    for n in indexes
        #Print iteration number
        print("\rStep number: $n")

        #Initialise velocity
        p = (I - z*z')randn(d+1)

        #Take RATTLE step
        (zprime, pprime, gradprime) = Rattle(z -> gradlogf(z; sigma=sigma, mu=mu), z, p, h, L; gradz = gradz)

        #Compute log-acceptance probability, based on Hamiltonian of the dynamics
        a = 1/2*p'*p - gradz - 1/2*pprime'*pprime + gradprime

        u = log(rand(Float64)) #Simulate from uniform to accept/reject

        if u < a #Accept proposal
            #Update position in both Euclidean and Stereographic space
            z = zprime
            x = SP(z; sigma=sigma, mu=mu)

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