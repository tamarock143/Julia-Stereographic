### We code up our version of a HMC algorithm in order to compare with our algorithms ###

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
HMC = function (logf, gradlogf, x0, N, delta, L; M = I(length(x0)), includefirst = true, steps = 1, printing = false)
    d = length(x0) #The dimension

    Minv = inv(M) #Invert M preemptively
    Msqrt = sqrt(M) #Sqrt M preemptively
    
    xout = zeros(N,d) #Prepare output

    #Slightly convoluted method for not storing the initial value WITHOUT allocating memory for an entirely new matrix
    if includefirst
        #If we want to include the first value, initialise the outputs
        indexes = 2:N
        xout[1,:] .= x0
    else
        #If we don't, start the indexes to be inputted at 1
        indexes = 1:N
    end

    x = x0 #Position vector, initialised at x0
    p = zeros(d) #Velocity vector, will be reinitialised according to Normal(0,M) at each step

    aout = 0 # Track acceptance rate
    
    for n in indexes
        #Print iteration number
        printing && print("\rStep number: $n")

        #We only sample one point after several steps
        for _ in 1:steps
            #Initialise velocity
            d > 1 ? p = Msqrt*randn(d) : p = Msqrt*randn()

            #Apply Leapfrog integrator to get proposals
            (xprime, pprime) = LeapFrog(gradlogf, x, p, delta, L; Minv = Minv)

            #Compute acceptance probability
            a = -logf(x) + logf(xprime) + (p'*Minv*p - pprime'*Minv*pprime)/2

            u = log(rand(Float64)) #Simulate from uniform to accept/reject

            if u < a #Accept proposal
                x = xprime #Update position
                aout += 1/(N*steps-1)
            end
        end

        #Add to output
        xout[n,:] .= x
    end
    println()

    return (x = xout, a = aout)
end

#### No U-turn

# An Julia Implementation of Efficient No-U-Turn Sampler described in Algorithm 3 in Hoffman et al. (2011)
# Author: Kai Xu
# Date: 06/10/2016

function eff_NUTS(θ0, ϵ, L, M; ∇L = missing, Δ_max = 1000, steps = 1)
  
    #- θ0      : initial model parameter
    #- ϵ       : leapfrog step size
    #- L       : likelihood function
    #- M       : sample number
  

  function leapfrog(θ, r, ϵ)
    #  - θ : model parameter
    #  - r : momentum variable
    #  - ϵ : leapfrog step size
    
    r̃ = r + (ϵ / 2) * ∇L(θ)
    θ̃ = θ + ϵ * r̃
    r̃ = r̃ + (ϵ / 2) * ∇L(θ̃)
    return θ̃, r̃
  end

  function build_tree(θ, r, u, v, j, ϵ, Δ_max)
    #- θ   : model parameter
    #- r   : momentum variable
    #- u   : log of slice variable
    #- v   : direction ∈ {-1, 1}
    #- j   : depth
    #- ϵ   : leapfrog step size
    
    if j == 0
      # Base case - take one leapfrog step in the direction v.
      θ′, r′ = leapfrog(θ, r, v * ϵ)
      n′ = u <= L(θ′) - 0.5 * dot(r′, r′)
      s′ = u < Δ_max + L(θ′) - 0.5 * dot(r′, r′)
      return θ′, r′, θ′, r′, θ′, n′, s′
    else
      # Recursion - build the left and right subtrees.
      θm, rm, θp, rp, θ′, n′, s′ = build_tree(θ, r, u, v, j - 1, ϵ, Δ_max)
      if s′ == 1
        if v == -1
          θm, rm, _, _, θ′′, n′′, s′′ = build_tree(θm, rm, u, v, j - 1, ϵ, Δ_max)
        else
          _, _, θp, rp, θ′′, n′′, s′′ = build_tree(θp, rp, u, v, j - 1, ϵ, Δ_max)
        end
        if rand() < n′′ / (n′ + n′′)
          θ′ = θ′′
        end
        s′ = s′′ & (dot(θp - θm, rm) >= 0) & (dot(θp - θm, rp) >= 0)
        n′ = n′ + n′′
      end
      return θm, rm, θp, rp, θ′, n′, s′
    end
  end

  ismissing(∇L) && (∇L = θ -> ForwardDiff.gradient(L, θ))  # generate gradient function

  θs = zeros(M, length(θ0))  # store samples
  θs[1,:] = θ0

  println("[eff_NUTS] start sampling for $M samples ($steps steps each) with ϵ=$ϵ")

  for m = 1:M-1
    theta = θs[m,:]
    thetaprime = zeros(length(theta))

    for _ in 1:steps #Iterate over number of steps
        r0 = randn(length(θ0))
        u = log(rand()) + L(theta) - 0.5 * dot(r0, r0) # Note: θ^{m-1} in the paper corresponds to
                                                    #       `theta` in the code
        θm, θp, rm, rp, j, thetaprime, n, s = theta, theta, r0, r0, 0, theta, 1, 1
        while s == 1
        v_j = rand([-1, 1]) # Note: this variable actually does not depend on j;
                            #       it is set as `v_j` just to be consistent to the paper
        if v_j == -1
            θm, rm, _, _, θ′, n′, s′ = build_tree(θm, rm, u, v_j, j, ϵ, Δ_max)
        else
            _, _, θp, rp, θ′, n′, s′ = build_tree(θp, rp, u, v_j, j, ϵ, Δ_max)
        end
        if s′ == 1
            if rand() < min(1, n′ / n)
            thetaprime = θ′
            end
        end
        n = n + n′
        s = s′ & (dot(θp - θm, rm) >= 0) & (dot(θp - θm, rp) >= 0)
        j = j + 1
        end

        theta = thetaprime
    end

    θs[m+1,:] = theta
  end

  println()
  println("[eff_NUTS] sampling complete")

  return (x = θs)
end