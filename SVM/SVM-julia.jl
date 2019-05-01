
using Random
using Gadfly
using LinearAlgebra
using DataFrames

hinge_loss(x :: Float64 ) = max(0,1-x)

mutable struct HyperPlan{d}
    
    w :: Vector{Float64}
    b :: Float64
    
    function HyperPlan{d}() where {d}
        new(zeros(d), 0.0)
    end
    
    function HyperPlan{d}(w :: Vector{Float64}) where {d}
        n = length(w)
        if d >= n
            w = [w;zeros(d-n)]
        else
            w = w[1:d]
        end
        new(w, 0.0)
    end   
    
    function HyperPlan{d}(w :: Vector{Float64}, 
                          b :: Float64) where {d}
        n = length(w)
        d >= n && (w = [w;zeros(d-n)])
        d >= n || (w = w[1:d])
        new(w, b)
    end
end

Base.ndims(::HyperPlan{d}) where {d} = d

function (h :: HyperPlan)( x :: Vector{Float64})
    dot(h.w, x) + h.b
end

import Base:+

function +(h1 :: HyperPlan{d}, h2 :: HyperPlan{d}) where {d}

    HyperPlan{d}(h1.w + h2.w, h1.b + h2.b)

end

import Base:-

function -(h1 :: HyperPlan{d}, h2 :: HyperPlan{d}) where {d}

    HyperPlan{d}(h1.w - h2.w, h1.b - h2.b)

end

function external_prod(h :: HyperPlan{d}, 
                       a :: Float64) where {d} 
    
    HyperPlan{d}(h.w .* a, h.b * a)
    
end

square(h :: HyperPlan) = dot(h.w, h.w) + h.b^2

function gradient_loss(h     :: HyperPlan{d}, 
                       x     :: Vector{Float64}, 
                       y     :: Float64, 
                       lambd :: Float64) where {d}
    if y * h(x) < 1
        v1 = lambd .* h.w .- y .* x
        v2 = lambd * h.b - y
    else
        v1 = lambd .* h.w
        v2 = lambd * h.b
    end
    HyperPlan{d}(v1, v2)
end

function SGD(h     :: HyperPlan{d}, 
             x     :: Array{Float64,2}, 
             y     :: Vector{Float64}, 
             lambd :: Float64,  
             niter     = 1000, 
             learning  = 0.1, 
             averaging = 1) where {d}
    
    i0       = abs(niter-averaging)
    liste    = HyperPlan{d}[]
    push!(liste, h)
    res0, res1 = h, h
    n = length(y)
    for i in 1:niter
        it = rand(1:n)
        vt = gradient_loss(res0, x[it,:], y[it], lambd)
        res0 = res0 - external_prod(vt,learning/(1+learning*lambd*i))
        push!(liste, res0)
        mut = 1/max(1,i-i0)
        res1 = (external_prod(res1, 1-mut)
              + external_prod(res0, mut))
        res0 = res1
    end
    res1, liste
end

using LinearAlgebra

mutable struct Polynomial
    
    weights :: Array{Float64, 2}
    void    :: Array{Float64, 2}

    function Polynomial(w)
        d = size(w)[1]
        void = zeros(Float64, (d,d))
        for i in 1:d
            for j in 1:d-i+1
                void[i,j] = 1
            end
        end
        weights = void .* w
        new( weights, void)
    end
end

degree(p :: Polynomial) = size(p.weights)[1]-1
    
function (p :: Polynomial)( x :: Vector{Float64} )
    @assert length(x) > 1
    X1 = [x[1]^exp for exp in 0:degree(p)]
    X2 = [x[2]^exp for exp in 0:degree(p)]
    ((p.weights * X1)' * X2)[1,1]
end

# Les paramètres :
# dimension de l'espace des x
h = HyperPlan{2}()
# degré du polynôme de départ
deg = 1
# on introduit ce petit pourcentage d'erreurs dans les données
seuil_erreur = 0.
# taille de l'échantillon de données d'entraînement
m=100
# taille de l'échantillon de données de test 
mtest=1000
# bornes sur l'espace des données x
xmax =  1.25
xmin = -1.25
# valeur de lambda pour le soft-SVM
lambd=1.
# valeur du taux d'apprentissage initialement pour le soft-SVM
learning  = 0.1
# taille du stencil pour l'averaging dans la SGD
averaging = 10

# fabrication des données
Random.seed!(1234)
w   = reshape(collect(0:(deg+1)*(deg+1)-1),deg+1,deg+1)
w[1,1] = 0.0

pf = Polynomial(w)
#x  = xmin .+ (xmax .- xmin) .* rand(Float64,(m, ndims(h)))
x1 = collect(range(xmin, stop=xmax, length=10))  .* ones(10)'
x2 = collect(range(xmin, stop=xmax, length=10))' .* ones(10)
x  = hcat(vec(x1),vec(x2))

u  = rand(Float64,m)
y  = zeros(m)
y[ u .>  seuil_erreur ] .=   sign.([pf(x[i,:]) for i in 1:m if u[i] >   seuil_erreur])
y[ u .<= seuil_erreur ] .= - sign.([pf(x[i,:]) for i in 1:m if u[i] <=  seuil_erreur])

plot(x = x[:,1], y = x[:,2], color = y)

Random.seed!(5678)
xtest = xmin .+ (xmax - xmin) .* rand(Float64,( mtest, ndims(h)))
ytest = sign.([pf(xtest[i,:]) for i in 1:mtest])
plot(x = xtest[:,1], y = xtest[:,2], color = ytest)

# Résolution du problème soft-SVM

h, liste = SGD(h, x, y, lambd, 200, learning, averaging)

h

function affiche(x, y, h :: HyperPlan)
    m = length(y)
    yresult = [sign(h(x[i,:])) for i in 1:m]
    
    p1 = plot( x = x[:,1], y = x[:,2], color=y)
    p2 = plot( x = x[:,1], y = x[:,2], color=yresult)
    
    hstack(p1, p2)
end

affiche(x, y, h)

affiche(xtest,ytest,h)

function loss(h :: HyperPlan{d}, 
     x :: Vector{Float64}, 
     y :: Float64) where {d} 
    
    hinge_loss(y * h(x))
end

function risk(h     :: HyperPlan{d}, 
              x     :: Array{Float64, d}, 
              y     :: Array{Float64, 1}, 
              lambd :: Float64) where {d}
    
    (lambd/2 * square(h)
        +1/length(y)*sum([loss(h, x[i,:], y[i]) 
                for i in 1:size(x)[1]]))
end 

function risk_vrai(h :: HyperPlan{d}, 
                   x :: Array{Float64,d}, 
                   y :: Vector{Float64}) where {d}
    n = length(y)
    @assert n == size(x)[1]
    sum([sign.(h(x[i,:])) != y[i] for i in 1:n])/n
end

function affiche_risques(liste, x, y, lambd)
    y1 = [risk(h, x, y, lambd) for h in liste]
    y2 = [risk_vrai(h, x, y) for h in liste]
    df1 = DataFrame( x = collect(1:length(liste)), 
        y = y1, label = :risque)
    df2 = DataFrame( x = collect(1:length(liste)), 
        y = y2, label = :risque_vrai)
    df = vcat(df1,df2)
    plot(df, x=:x, y=:y, Geom.line, 
        color= :label, 
        Scale.color_discrete_manual("blue","red"))
end

affiche_risques(liste,xtest,ytest,lambd)


