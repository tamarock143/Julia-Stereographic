### Sub Cauchy testing

    include("Adaptive SBPS.jl")
    include("Hamiltonian MC.jl")
    include("Adaptive SRW.jl")
    include("SHMC.jl")
    include("Adaptive Slice.jl")


    using ForwardDiff
    using Plots
    using SpecialFunctions
    using StatsBase
    using JLD

#### ADAM Testing
    d = 10
    lat = 2

    f = function(z, theta::AbstractVector{T}) where T
        d = length(z)-1

        tnorm = sum(theta[d+1:2d].^2)

        mid = theta[d+1:2d]/sqrt(tnorm + 1e-2)*tanh(tnorm/2)*(1-(lat-1)^2)
        #mid = zeros(d)

        A = UpperTriangular(zeros(T,d,d))
        k=2d+1
        for i in 1:d
            for j in i:d
                A[i,j] += theta[k]
                k += 1
            end
        end

        sigma=Symmetric(transpose(A)*A + 1e-3*I(d))

        (x,M) = SubC(z; sigma = sigma, mu = theta[1:d], obs = vcat(mid,lat), jacobian=true)

        return -log(M) - logf(x)
    end

    z = unifsim(100000, d; obs = lat)

    logf(x) = -d*log(d + sum(x.^2))

    theta = randn(Int64(2d+d*(d+1)/2))

    out = ADAM(f, z, theta, 1; N = 100000)

    
    A = UpperTriangular(zeros(T,d,d))
    k=d+1
    for i in 1:d
        for j in i:d
            A[i,j] += out[k]
            k += 1
        end
    end

    sigma=Symmetric(transpose(A)*A + 1e-3*I(d))

    plot(eigen(sigma).values)
    hline!([sqrt(d)/2])

    x = zeros(10000,10)
    for k in 1:10000
        x[k,:] = SubC(z[k,:]; sigma = sigma, mu = out[1:d], obs = vcat(zeros(d),lat), jacobian=false)
    end

#### Skewness Simulating

d = 2
alpha = 5

skewsim = function(n; d = d, alpha = alpha)
    out = zeros(n,d)
    for i in 1:n
        y = randn(d)
        p = 1 ./ (1 .+ exp.(-alpha .* y))

        flips = (-1) .^ (rand(d) .> p)

        out[i,:] = flips .* y
    end

    return out
end