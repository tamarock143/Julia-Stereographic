#We code up our version of a HMC algorithm in order to compare with our SBPS

#Import LinearAlgebra library
using LinearAlgebra

#Leapfrog Integrator: take L leapfrog steps of length delta
LeapFrog = function (gradlogf, x, p, delta, L; Minv = I(length(x)))
    p += delta/2 * gradlogf(x) #First "half-update" for p

    #We treat the final leapfrog separately, since it only needs a half-update for p
    for i in 1:L-1
        x += delta*Minv*p #Update x
        p += delta*gradlogf(x) #Update p
    end

    x += delta*Minv*p #Update x
    p += delta/2*gradlogf(x) #Final "half-update" for p

    return (x = x, p = p)
end

#HMC Algorithm
HMC = function (logf, gradlogf, x0, N, delta, L; M = I(length(x0)))
    d = length(x0) #The dimension

    Minv = inv(M) #Invert M preemptively
    Msqrt = sqrt(M) #Sqrt M preemptively

    xout = zeros(N,d) #Prepare output
    xout[1,:] = x0

    x = x0 #Position vector, initialised at x0
    p = zeros(d) #Velocity vector, will be reinitialised according to Normal(0,M) at each step

    for n in 2:N
        #Print iteration number
        print("\rStep number: $n")

        #Initialise velocity
        p = Msqrt*randn(d)

        #Apply Leapfrog integrator to get proposals
        (xprime, pprime) = LeapFrog(gradlogf, x, p, delta, L; Minv = Minv)

        #Compute acceptance probability
        a = -logf(x) + logf(xprime) + (p'*Minv*p - pprime'*Minv*pprime)/2

        u = log(rand(Float64)) #Simulate from uniform to accept/reject

        if u < a #Accept proposal
            xout[n,:] = xprime
            x = xprime #Update position
        else #Reject proposal
            xout[n,:] = x
        end
    end
    println()

    return (x = xout)
end