#Attempt at coding up a stereographic HMC

#Import Stereographic Projection stuff, and the Random library
include("Stereographic Projection.jl")
using Random

#RATTLE integrator for Spherical constraint: take L RATTLE steps of length h
Rattle = function (gradlogf, z, p, h, L)
    #Check that norm(z) == 1 and z.p=0
    abs(sum(z.^2) -1) >= 1e-12 && error("norm(z) != 1")
    abs(sum(z.*p)) >= 1e-12 && error("z.p != 0")

    #Initialise gradient and product
    gradz = gradlogf(z)
    graddotz = sum(gradz.*z)

    for i in 1:L
        #Initialise the Lagrange multiplier lambda
        lambda = (h^2/2*graddotz + 1 + sqrt(h^4/2*graddotz^2 - h^2*graddotz + 1 - h^4/2*sum(gradz.^2) - 2*h^3*sum(p.*gradz) - 2*h*sum(p.^2)))/h^2

        p += h/2*(gradz - lambda*z) #First "half-update" for p

        z += h*p #Update z

        #Check that norm(z) == 1
        abs(sum(z.^2) -1) >= 1e-12 && error("norm(z) != 1")

        #Update gradient and product
        gradz = gradlogf(z)
        graddotz = sum(gradz.*z)

        #Second "half-update" for p
        p += h/2*gradz
        p *= I(d) - z*z'
    end

    return(z = z, p = p)
end