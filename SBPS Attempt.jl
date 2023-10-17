#Import Stereographic Projection stuff, and the Random library
include("Stereographic Projection.jl")
using Random
using Plots
using SpecialFunctions

## SBPS Path Simulator

#We start by simulating a multivariate student's t-distribution with d degrees of freedom
#This is the easiest distribution to target with SBPS, since the density is uniform on the sphere
#This means there will be no bounce events
SBPSStudentSimulator = function (x0, lambda, T, delta)
    z = SPinv(x0) #Map to the sphere
    v = SBPSRefresh(z) #Initialize velocity
    d = length(x0) #The dimension

    n = floor(BigInt, T/delta)+1 #Total number of observations of the skeleton path

    #Prepare output
    zout = zeros(n,d+1)
    vout = zeros(n,d+1)

    zout[1,:] = z
    vout[1,:] = v

    left::Float64 = (n-1)delta #Track remaining amount of time left until last observation
    k = 2 #Track the next row to be added to output
    t0::Float64 = delta #Amount of time after an event until the next skeleton path sample time

    while left > 0
        #Simulate next refreshment time according to Exp(lambda)
        tauref = randexp(Float64)/lambda

        t = min(tauref,left) #In case the path ends before the next event
        
        #Update remaining path length
        left -= t 

        #If the next observation time is after the event time, we do not need to simulate the path
        if t0 > t
            #Update next observation time
            t0 -= t
        else
            #Simulate the next piece of the path
            (zpath,vpath) = SBPSDetPath(z,v,t,delta; t0)

            #Number of rows to add.
            nadd = size(zpath)[1]

            k + nadd -1 > n && error("something strange happened with adding rows")

            #Append this piece to the output
            zout[k:k+nadd-1,:] = zpath
            vout[k:k+nadd-1,:] = vpath

            #Increment next row to add
            k += nadd

            #Time to next observation time
            t0 += (floor((t-t0)/delta)+1)delta -t
        end
        
        #Update position and velocity based on whether we had a refreshment event
        if tauref <= left
            z = cos(t)z + sin(t)v
            v = SBPSRefresh(z)
        else
            (z,v) = (cos(t)z + sin(t)v, cos(t)v - sin(t)z)
        end
    end

    #Ensure we have added final position and velocity
    zout[n,:] = z
    vout[n,:] = v

    return (z = zout, v = vout)
end

#First attempt:

#Define parameters etc
d=50
x0 = randn(d)
lambda = 1
T = 10000
delta = 0.1

#Run the simulation
(z,v) = SBPSStudentSimulator(x0,lambda,T,delta)

#Prepare the projected output on R^d
n = floor(BigInt,T/delta)+1
x = zeros(n,d)

#Project each entry back to R^d
for i in 1:n
    x[i,:] = SP(z[i,:])
end

#Plot comparison against the true distribution
p(x) = gamma((d+1)/2)/(sqrt(d*pi)*gamma(d/2))*(1+x^2/d)^-((d+1)/2)
b_range = range(-5, 5, length=51)

histogram(x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
plot!(p,label="Analytical", lw=3, color=:red) #Idk why this is flagging as possible error
xlims!(-5, 5)
ylims!(0, 0.4)
title!("")
xlabel!("x")
ylabel!("P(x)")