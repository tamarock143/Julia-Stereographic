#Brent's Method for 1 dimensional function minimisation
#We need the parabolic interpolation method, and the Golden search method

ParaSearch = function (f,a,b,c; fa = missing, fb = missing, fc = missing)
    #If not provided, evaluate f at each point
    isequal(fa,missing) && (fa = f(a))
    isequal(fb,missing) && (fb = f(b))
    isequal(fc,missing) && (fc = f(c))

    #Ensure we have suitable input
    (!(a<b<c) || fb > fa || fb > fc) && error("ParaSearch input error")

    #Parabolic interpolation
    x = b - 1/2 *((b-a)^2*(fb - fc) - (b-c)^2*(fb - fa))/((b-a)*(fb - fc) - (b-c)*(fb - fa))
    fx = f(x)

    if x <= b
        fx <= fb ? ((a,fa),(x,fx),(b,fb)) : ((x,fx),(b,fb),(c,fc))
    else
        fx <= fb ? ((b,fb),(x,fx),(c,fc)) : ((a,fa),(b,fb),(x,fx))
    end
end

GoldenSearch = function (f,a,b,c; fa = missing, fb = missing, fc = missing)
    #If not provided, evaluate f at each point
    isequal(fa,missing) && (fa = f(a))
    isequal(fb,missing) && (fb = f(b))
    isequal(fc,missing) && (fc = f(c))
    
    #Ensure we have suitable input
    (!(a<b<c) || fb > fa || fb > fc) && error("GoldenSearch input error")

    #Find Golden Search interpolation point
    x = c+a-b
    fx = f(x)

    if x <= b
        fx <= fb ? ((a,fa),(x,fx),(b,fb)) : ((x,fx),(b,fb),(c,fc))
    else
        fx <= fb ? ((b,fb),(x,fx),(c,fc)) : ((a,fa),(b,fb),(x,fx))
    end
end

#We will initialise Brent's method by performing a grid search over the interval
GridSearch = function (f,a,b,n)
    mygrid = Array{Float64}(undef, (n+2,2))

    for i in 1:n+2
        x=a+(b-a)*(i-1)/(n+1)
        mygrid[i,:] = [x,f(x)]
    end

    return mygrid
end

Brent = function (f,a,b)
    fa = f(a)
    fb = f(b)


end
