#Brent's Method for 1 dimensional function minimisation
#We need the parabolic interpolation method, and the Golden search method

ParaSearch = function (f,a,b,c; fa = missing, fb = missing, fc = missing, triplet = false)
    #ParaSearch includes many fail cases, which we account for going into Brent's Method
    #Do we have a repeated point?
    !(a != b != c) && return "Fail"

    #If not provided, evaluate f at each point
    isequal(fa,missing) && (fa = f(a))
    isequal(fb,missing) && (fb = f(b))
    isequal(fc,missing) && (fc = f(c))

    #Check points are well ordered
    !(a<b<c) && ((a,fa),(b,fb),(c,fc)) = sort([(a,fa),(b,fb),(c,fc)], by = x -> x[1])

    #Ensure we have suitable input
    (fb > fa || fb > fc) && return "Fail"

    #Ensure we do not divide by 0 (tests whether points are colinear)
    (b-a)*(fb - fc) == (b-c)*(fb - fa) && return "Fail"

    #Parabolic interpolation
    x = b - 1/2 *((b-a)^2*(fb - fc) - (b-c)^2*(fb - fa))/((b-a)*(fb - fc) - (b-c)*(fb - fa))
    fx = f(x)

    if triplet #Do we output the new bracketing interval, or just the new point?
        if x <= b
            fx <= fb ? ((a,fa),(x,fx),(b,fb)) : ((x,fx),(b,fb),(c,fc))
        else
            fx <= fb ? ((b,fb),(x,fx),(c,fc)) : ((a,fa),(b,fb),(x,fx))
        end
    else
        (x,fx)
    end
    
end

GoldenSearch = function (f,a,b,c; fa = missing, fb = missing, fc = missing, triplet = false)
    #If not provided, evaluate f at each point
    isequal(fa,missing) && (fa = f(a))
    isequal(fb,missing) && (fb = f(b))
    isequal(fc,missing) && (fc = f(c))
    
    #Check points are well ordered
    !(a<b<c) && ((a,fa),(b,fb),(c,fc)) = sort([(a,fa),(b,fb),(c,fc)], by = x -> x[1])

    #Ensure we have suitable input
    (fb > fa || fb > fc) && return "No min"

    w = (3-sqrt(5))/2 #Golden ratio

    #Find Golden Search interpolation point
    if c-b > b-a #Check whether longer interval is to the left or right of b
        x = b + (c-b)w
        fx = f(x)

        if triplet #Do we output the new bracketing interval, or just the new point?
            fx <= fb ? ((b,fb),(x,fx),(c,fc)) : ((a,fa),(b,fb),(x,fx))
        else
            (x,fx)
        end
    else
        x = b - (b-a)w
        fx = f(x)

        if triplet #Do we output the new bracketing interval, or just the new point?
            fx <= fb ? ((a,fa),(x,fx),(b,fb)) : ((x,fx),(b,fb),(c,fc))
        else
            (x,fx)
        end
    end
end

#We will initialise Brent's method by performing a grid search over the interval
#No longer using GridSearch in Brent. Have replaced with simple Binary search
GridSearch = function (f,a,c,n,tol)
    mygrid = Array{Tuple{Float64,Float64}}(undef,n+2)

    for i in 1:n+2
        x = a+(c-a)*(i-1)/(n+1)
        mygrid[i] = (x,f(x))
    end

    return mygrid
end

Brent = function (f,a,c,tol)
    #Start by performing a Binary search, by looking at the midpoint of the interval
    #If we find the minimum to be at an endpoint, we refine the search
    looking = true
    fa = f(a)
    fc = f(c)

    while looking && c-a > tol
        b = (c+a)/2
        fb = f(b)

        #If the minimum is at an endpoint, focus search to be near that endpoint
        if fa > fb < fc
            looking = false
        elseif fa < fc
            (c,fc) = (b,fb)
        else
            (a,fa) = (b,fb)
        end
    end

    #If the Binary search chose an endpoint as the min, return that endpoint
    if looking
        return sort([(a,fa),(c,fc)], by = x -> x[2])[1]
    end

    #Initialise alternations between GoldenSearch and ParaSearch. These variables are required for bookkeeping
    x = w = v = b
    fx = fw = fv = fb

    while c-a > tol
        
    end
end
