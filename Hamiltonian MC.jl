#We code up our version of a HMC algorithm in order to compare with our SBPS

#Import LinearAlgebra library
using LinearAlgebra

HMC = function (gradlogf, x0, N::BigInt, deltat, L; M = I(length(x0)))
    d = length(x0) #The dimension
    xout = zeros(N,d) #Prepare output

    p = sqrt(M)*randn(d) #Velocity vector, follows Normal(0,M)


end