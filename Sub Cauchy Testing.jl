### Sub Cauchy testing

    include("Adaptive SBPS.jl")
    include("Hamiltonian MC.jl")
    include("Adaptive SRW.jl")
    include("SHMC.jl")
    include("Adaptive Slice.jl")
    include("SubCauchy Sampling.jl")


    using ForwardDiff
    using Plots
    using SpecialFunctions
    using StatsBase
    using JLD

    log1exp = function(x)
        if x > 33
            return x
        elseif x < -33
            return exp(x)
        else
            return log(1+exp(x))
        end
    end

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

    sigma = sqrt(d)/2*I(d)
    mu = zeros(d)
    lat = 1.5

    obs = vcat(zeros(d), lat)


    z = unifsim(200000, d; obs = lat)

    x = zeros(200000,d)
    for k in 1:200000
        x[k,:] = SubC(z[k,:]; sigma = sigma, mu = mu, obs = obs, jacobian=false)
    end

    histogram2d(x[:,1],x[:,2],bins=(range(-2, 2, length=51), range(-2, 2, length=51)), normalize=:pdf, color=:inferno)

##### SCSS tests
    d = 50
    nu = 50
    alpha = 15

    #logf = x -> -sum(x.^2)/2
    logf = x -> -(nu+d)/2*log(nu + sum(x.^2)) - log1exp(-alpha*x[1])

    #lat = 2

    obj = function(z, theta::AbstractVector{T}) where T
        d = length(z)-1

        lat = (1+tanh(theta[2d+1]))/2

        tnorm = sum(theta[d+1:2d].^2)

        mid = theta[d+1:2d]/sqrt(tnorm + 1e-2)*tanh(tnorm/2)*(1-(lat-1)^2)
        #mid = zeros(d)

        #A = zeros(T,d,d)
        #k=2d+1
        #for i in 1:d
        #    for j in i:d
        #        A[i,j] += theta[k]
        #        k += 1
        #    end
        #end

        #sigma= Symmetric(A * transpose(A)) + 1e-3*I(d)

        sigma = diagm((exp.(theta[d+1:2d]) .+ 1e-3))

        (x,M) = SubC(z; sigma = sigma, mu = theta[1:d], obs = vcat(mid,lat), jacobian=true)

        return -log(M) - logf(x)
    end
    
    z = unifsim(10000, d; obs = lat- 1e-2)

    theta = randn(2d)#vcat(zeros(2d),randn(Int64(d*(d+1)//2)))
    
    @time out = ADAM(obj, z, theta, 1; N = 100000)
    save("myout.jld","myout",out)
    out = load("myout.jld")["myout"]

    mu = out[1:d]
    
    #tnorm = sum(out[d+1:2d].^2)
    #mid = out[d+1:2d]/sqrt(tnorm + 1e-2)*tanh(tnorm/2)*(1-(lat-1)^2)
    mid=zeros(d)
    obs = vcat(mid,lat)

    #A = zeros(d,d)
    #k=2d+1
    #for i in 1:d
    #    for j in i:d
    #        A[i,j] += theta[k]
     #       k += 1
      #  end
    #end

    #sigma= Symmetric(A * transpose(A)) + 1e-3*I(d)
    sigma= diagm((exp.(out[d+1:2d]) .+ 1e-3))

    plot(eigen(sigma).values)
    hline!([sqrt(d)/2])

    x0 = randn(d)
    
    N = 200000

    @time outSCS = SubCauchySlice(logf, x0, N; sigma = sigma, mu = mu, obs = obs);
    save("outSCS.jld","outSCS",outSCS)

    outSCS = load("outSCS.jld")["outSCS"]
    
    @time outSSS = SliceSimulator(logf, x0, N; sigma = sigma, mu = mu);
    save("outSSS.jld","outSSS",outSSS)
    
    outESS = EllipticalSliceSimulator(logf, x0, N; sigma = 2*sigma/sqrt(d), mu = mu);

    #Plot comparison against the true distribution
    p(x) = 1/sqrt(2pi)*exp(-x^2/2)
    #q(x) = 1/sqrt(2pi*sigmaf)*exp(-x^2/2sigmaf)
    q(x) = gamma((nu+1)/2)/(sqrt(nu*pi)*gamma(nu/2))*(1+x^2/nu)^-((nu+1)/2)
    b_range = range(-10,10, length=101)

    histogram(out.x[:,1], label="Experimental", bins=b_range, normalize=:pdf, color=:gray)
    plot!(p, label= "N(0,1)", lw=3)
    plot!(q, label= "t", lw=3)
    xlabel!("x")
    ylabel!("P(x)")


    plot(1:N,outSCS.x[:,1], label = "SCS")
    plot(1:N,outSSS.x[:,1], label = "SSS")
    plot(1:N,outESS.x[:,1], label = "ESS")

    plot(1:N,sum(x -> x^2, outSCS.x, dims=2), label = "SCS", ylims = [20,160])
    plot(1:N,sum(x -> x^2, outSSS.x, dims=2), label = "SSS", ylims = [20,160])
    plot(1:N,sum(x -> x^2, outESS.x, dims=2), label = "ESS")

    histogram2d(outSCS.x[:,1],outSCS.x[:,2], bins=(1000,1000),normalize=:pdf)
    histogram2d(outSSS.x[:,1],outSSS.x[:,2], bins=(1000,1000),normalize=:pdf)
    