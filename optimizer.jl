using Optim
using LinearAlgebra
using IterTools

function multen(n)
    d = n^2
    f(k) = (rep = zeros(n,n); rep[1+(k-1)÷n, 1+(k-1)%n] = 1; rep)
    [f(k) == f(i)*f(j) ? 1. : 0. for (k,i,j) in product(1:d,1:d,1:d)]
end

function J0(v, d, m, c)
    tmp = reshape(v, (d,m,3))
    r, a, b = tmp[:,:,1]', tmp[:,:,2]', tmp[:,:,3]'
    X = [c[j,k,l] - sum(r[μ,j]*a[μ,k]*b[μ,l] for μ in 1:m)
        for (j,k,l) in product(1:d,1:d,1:d)]
    norm(X)^2
end

function run(d, m; v0 = randn(3*d*m), c=multen(Int(sqrt(d))),
                    optimizer=BFGS(), shw=false, it=1000)
    optimize(v -> J0(v, d, m, c), v0, optimizer,
                    Optim.Options(show_trace=shw, iterations=it))
end

function run2(d, m; c=multen(Int(sqrt(d))),
                    optimizer=BFGS(), shw=false,
                    it=1000, it2=10000)
    succ = false
    x=0
    while succ == false
        x = run(d, m; v0=randn(3*d*m), c=c,
                optimizer=optimizer, shw=shw, it=it)
        println(x.minimum)
        succ = (x.iterations == it ? false : true)
        if x.minimum < 1
            x = run(d, m; v0=x.minimizer, c=c,
                optimizer=optimizer, shw=shw, it=it2)
            println(x.minimum)
            succ = (x.iterations == it2 ? false : true)
        end
    end
    return x
end
