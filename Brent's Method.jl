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
    mygrid = Array{Tuple{Float64,Float64}}(undef,n+2)

    for i in 1:n+2
        x = a+(b-a)*(i-1)/(n+1)
        mygrid[i] = (x,f(x))
    end

    return mygrid
end

Brent = function (f,a,b)
    #Start by performing a GridSearch, initialised to output 4 points
    #If we find the minimum to be at an endpoint, we refine the search
    looking = true

    while looking && b-a > 10^-6
        #We sort the GridSearch based on f(x)
        mygrid = sort(GridSearch(f,a,b,2), by = x -> x[2])

        #If the minimum is at an endpoint, focus search to be near that endpoint
        if mygrid[1][1] == a
            b = a + (b-a)/3
        elseif mygrid[1][1] == b
            a = b - (b-a)/3
        else
            looking = false
        end
    end

end
