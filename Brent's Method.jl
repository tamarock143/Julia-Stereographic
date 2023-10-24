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

    #Ensure we have suitable input to search for a minimum
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

Brent = function (f,a,b,tol)
    #Start by performing a Binary search, by looking at the midpoint of the interval
    #If we find the minimum to be at an endpoint, we refine the search
    looking = true
    fa = f(a)
    fb = f(b)

    while looking && b-a > tol
        x = (b+a)/2
        fx = f(x)

        #If the minimum is at an endpoint, focus search to be near that endpoint
        if fb > fx < fb
            looking = false
        elseif fa < fb
            (b,fb) = (x,fx)
        else
            (a,fa) = (x,fx)
        end
    end

    #If the Binary search chose an endpoint as the min, return that endpoint
    if looking
        return sort([(a,fa),(b,fb)], by = x -> x[2])[1]
    end

    #Initialise alternations between GoldenSearch and ParaSearch. These variables are required for bookkeeping
    ((v,fv),(w,fw)) = sort([(a,fa),(b,fb)], by = x -> x[2])

    while b-a > tol
        #Attempt ParaSearch
        p = ParaSearch(f, x, w, v, fa = fx, fb = fw, fc = fv)

        #If ParaSearch did not yield satisfactory results, use GoldenSearch
        if p == "Fail" || !(a <= p[1] <= b) || abs(fx - p[2]) <= abs(fv - fw)/2
            (u,fu) = GoldenSearch(f, a, x, b, fa = fa, fb = fx, fc = fb)
        else
            (u,fu) = p
        end

        #Update variables
        (w,fw) = (v,fv)
        if fu <= fx
            (v,fv) = (x,fx)
            if u <= x
                ((a,fa),(x,fx),(b,fb)) = ((a,fa),(u,fu),(x,fx))
            else
                ((a,fa),(x,fx),(b,fb)) = ((x,fx),(u,fu),(b,fb))
            end
        else
            fu <= fv && (v,fv) = (u,fu)
            if u <= x
                (a,fa) = (u,fu)
            else
                (b,fb) = (u,fu)
            end
        end
    end
end
