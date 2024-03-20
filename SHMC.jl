#Attempt at coding up a stereographic HMC

#Import Stereographic Projection stuff, and the Random library
include("Stereographic Projection.jl")
using Random

#RATTLE integrator for Spherical constraint: take L RATTLE steps of length h
Rattle = function (gradlogf, z0, p0, h, L)
    #Check that norm(z) == 1 and z.p=0
    abs(sum(z0.^2) -1) >= 1e-12 && error("norm(z) != 1")
    abs(sum(z0.*p0)) >= 1e-12 && error("z.p != 0")

    #Initialise position and momentum variables
    z = z0
    p = p0

    #Initialise gradient and product
    gradz = gradlogf(z)
    graddotz = sum(gradz.*z)

    for i in 1:L
        #Check the discriminant to ensure lambda exists
        if (disc = graddotz + 4/h^4 - 4/h^2*sum(x -> x^2, p - h/2*gradz)) < 0
            #If we cannot solve the RATTLE equations, reject move and return to initial position
            return(z = z0, y = y0)
        end
        #Initialise the Lagrange multiplier lambda
        lambda = 2/h^2 + graddotz + sqrt(graddotz + 4/h^4 - 4/h^2*sum(x -> x^2, p - h/2*gradz))
        
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

    return(z = z, p = p)
end