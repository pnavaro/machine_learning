
using Plots, ProgressMeter, LinearAlgebra
gr()

m, d  = 1000, 2
coef3 = 2 * rand(d,d,d) .- 1
coef2 = 2 * rand(d,d) .- 1
coef1 = 2 * rand(d,1)
coef0 = rand(1)

struct Stump
    
    j :: Int64
    theta :: Float64
    b :: Float64
    
    function Stump(j, theta, b)
        @assert j > 0
        new(j, theta, b)
    end
    
end

function (self :: Stump)( x )
    self.b * sign.(self.theta .- x[:,self.j])
end
    
function erreur(self :: Stump, x, y, D)
    sum( D .* (y .!= self(x)))
end

mutable struct Hypotheses
    
    weights :: Vector{Float64}
    stumps  :: Array{Stump,1}
    D       :: Vector{Vector{Float64}}
    
    function Hypotheses( )
        weights = Float64[]
        stumps  = Stump[]
        D       = Vector{Float64}[]
        new( weights, stumps, D)
    end
    
end

function (self::Hypotheses)(x)
    T = length(self.weights)
    m = size(x)[1]
    s = zeros(Float64, m )
    for t in 1:T
        s .+= self.weights[t] .* self.stumps[t](x)
    end
    sign.(s)
end

function erreur(self,x,y)
    m = length(y)
    sum(self(x) .!= y)/m
end

function plot_results( x, y, ytest)
    
    p    = plot(layout=(1,2); )
    blue = y .< 0
    red  = y .> 0
    scatter!(p[1,1],x[red,1],x[red,2],label=:Xb, title=:test)
    scatter!(p[1,1],x[blue,1],x[blue,2],label=:or)
    blue = ytest .< 0
    red  = ytest .> 0
    scatter!(p[1,2],x[red,1],x[red,2],label=:Xb, title=:test)
    scatter!(p[1,2],x[blue,1],x[blue,2],label=:or)
    p
    
end

function ERM_Stump( x :: Array{Float64,2},
                    y :: Vector{Float64},
                    D :: Vector{Float64})
    m, d       = size(x)
    Fstar      = 1e15
    F          = Fstar
    bstar      = 1
    thetastar  = 0.
    jstar      = 1
    indice_tri = hcat([sortperm(x[:,j]) for j in 1:d]...) #np.argsort(x,axis=0)
    xsort      = zeros(Float64,m+1)
    for j in 1:d
        xsort[1:m] = x[indice_tri,j][:,j]
        xsort[end] = xsort[m]+1
        ysort = y[indice_tri[:,j]]
        Dsort = D[indice_tri[:,j]]
        F = sum(Dsort .* (ysort .+ 1)/2)
        if F < Fstar
            bstar = 1
            Fstar = F
            thetastar = xsort[1] - 1
            jstar = j
        end
        for i in 1:m
            F = F - ysort[i] * Dsort[i]
            if F < Fstar && xsort[i] < xsort[i+1]
                bstar=1
                Fstar=F
                thetastar=(xsort[i]+xsort[i+1])/2
                jstar=j
            end
        end
        F = sum(Dsort .* (1 .- ysort)/2)
        if F < Fstar
            bstar = -1
            Fstar = F
            thetastar = xsort[1]-1
            jstar = j
        end
        for i in 1:m
            F = F + ysort[i] * Dsort[i]
            if F < Fstar && xsort[i]<xsort[i+1]
                bstar = -1
                Fstar = F
                thetastar = (xsort[i]+xsort[i+1])/2
                jstar = j
            end
        end
    end
    Stump(jstar, thetastar, bstar)
end

function generate_data(m, d)
    x = 2 * rand(m,d) .- 1
    y = zeros(Float64,m)
    for i in 1:d, j in 1:d, k in 1:d
        y .+= coef3[i,j,k] .* x[:,i] .* x[:,j] .* x[:,k]
    end
    y = sign.(y .+ diag((x * coef2) * transpose(x)) .+ x * coef1 .+ coef0);
    x, vec(y)
end

x, y = generate_data(m, d)

#d = 2
#m = 1000
#f = open("test1.txt")
#data = [ transpose(map(x->parse(Float64,x),split(line," "))) for line in eachline(f)]
#data = vcat(data...)
#
#x = data[:,1:2]
#y = data[:,3]


D = 1 / m .* ones(Float64, m)
h = ERM_Stump(x,y,D)
plot_results( x, y, h(x))

function adaboost( x, y , T)
    
    m = size(x)[1]
    D = 1 / m * ones(Float64,m)
    H = Hypotheses()
    epsilon = Float64[]
    
    @showprogress 1 for t in 1:T
        ht = ERM_Stump(x, y, D)
        hx = ht(x)
        e  = erreur(ht, x, y, D)
        w0 = 0.5 * log(1 / e-1)
        push!(epsilon, e)
        push!(H.weights, w0)
        push!(H.stumps, ht)
        #push!(H.D, D)
        D = D .* exp.(-w0 * y .* hx)
        D = D / sum(D)
    end
    H, epsilon
end

H, epsilon = adaboost( x, y, 100)
yresult = H(x)
plot_results( x, y, yresult)

a=100*(1-erreur(H,x,y))
println("Performance sur les données de test :",a[1],"%")

xtest, ytest = generate_data(m, d)
plot_results( xtest, ytest, H(xtest))

a=100*(1-erreur(H,xtest,ytest))
println("Performance sur les données de test :",a[1],"%")

err       = Float64[]
err0      = zeros(Float64,m)
err_test  = Float64[]
err_test0 = zeros(Float64,m)
for (weight, stump)  in zip(H.weights, H.stumps)
    err0 .+= weight .* stump(x)
    push!(err, sum( sign.(err0) .!= y) / m)
    err_test0 .+= weight .* stump(xtest)
    push!(err_test, sum(sign.(err_test0) .!= ytest) / m)
end

plot(err; marker=:o)
plot!(err_test; marker=:s)
xlabel!("Itérations")
ylabel!("Erreur en %")

scatter(epsilon)
xlabel!("steps")
ylabel!("Epsilon")
ylims!(minimum(epsilon), 0.5)

f = open("data_banknote_authentication.txt")
databrute = [ transpose(map(x->parse(Float64,x),split(line,","))) for line in eachline(f)]
data = vcat(databrute...);

@show n = size(data)[1]
print("Pourcentage de vrais billets :",(n-sum(data[:,5]))/n*100,"%")

using Random
taille       = n
taille_train = trunc(Int64,taille*0.8)
taille_test  = taille-taille_train
indice       = randperm(taille)
train        = data[indice[1:taille_train],:]
test         = data[indice[taille_train+1:end],:]
xtrain       = train[:,1:4]
ytrain       = 2 * train[:,5] .- 1
xtest        = test[:,1:4]
ytest        = 2 * test[:,5] .- 1
d            = 4
m            = taille_train

H, epsilon = adaboost(xtrain, ytrain, 50)

yresult     =  H(xtrain)
yresulttest =  H(xtest);

a=100*(1 .- erreur(H,xtrain,ytrain))
print("Performance sur les données d'entraînement :",a[1],"%")

a=100*(1-erreur(H,xtest,ytest))
print("Performance sur les données de test :",a[1],"%")

nt        = length(H.stumps)
err       = Float64[]
err0      = zeros(Float64,length(ytrain))
err_test  = Float64[]
err_test0 = zeros(Float64,length(ytest))

for i in 1:nt
    err0 .+= H.weights[i] .* H.stumps[i](xtrain)
    push!(err, sum(sign.(err0) .!= ytrain) / length(ytrain))
    err_test0 .+= H.weights[i] .* H.stumps[i](xtest)
    push!(err_test, sum(sign.(err_test0) .!= ytest)/length(ytest))
end

plot(err)
plot!(err_test)
xlabel!("Itérations ")
ylabel!("Erreur en %")

plot(epsilon)
xlabel!("Itérations")
ylabel!("Epsilon")
ylims!(minimum(epsilon), 0.5)

print("borne d'erreur théorique ",exp(-2*sum((1/2 .- epsilon).^2))*100,"%")

plot_results(xtrain, ytrain, yresult)

plot_results(xtest, ytest, yresulttest)

f = open("EEG_Eye_State.txt")

databrute = [ transpose(map(x->parse(Float64,x),split(line,","))) for line in eachline(f)]
data = vcat(databrute...)

n = size(databrute)[1]
println("Nombre de données : ", n)

count = sum( data[:,15] .== 0)
println("Pourcentage de classe 0 :",count/n*100,"%")

n_train  = trunc(Int64,n/3)
n_valid  = trunc(Int64,n/3)
n_test   = n-n_train-n_valid
indice   = randperm(n)
train    = data[indice[1:n_train],:]
valid    = data[indice[n_train+1:n_train+n_valid],:]
test     = data[indice[n_train+n_valid+1:end],:]
x        = train[:,1:14]
y        = 2 * train[:,15] .- 1
xvalid   = valid[:,1:14]
yvalid   = 2 * valid[:,15] .- 1
xtest    = test[:,1:14]
ytest    = 2 * test[:,15] .- 1
d        = 14
m        = n_train

(H,epsilon)=adaboost(x,y,2000)

yresult     = H(x)
yresulttest = H(xtest)

a = sum(yresult .== y) / m * 100
println("Performance sur les données d'entraînement :",a[1],"%")

a = sum(yresulttest .== ytest) / n_test*100
println("Performance sur les données de test :",a[1],"%")

err        = Float64[]
err0       = zeros(Float64,length(y))
err_valid  = Float64[]
err_valid0 = zeros(Float64,length(yvalid))
err_test   = []
err_test0  = zeros(Float64,length(ytest))
nt = length(H.stumps)
for i in 1:nt
    err0 .+= H.weights[i] .* H.stumps[i](x)
    push!(err, sum(sign.(err0) .!= y) / length(y))
    err_test0 .+= H.weights[i] .* H.stumps[i](xtest)
    push!(err_test, sum(sign.(err_test0) .!= ytest) / length(ytest))
    err_valid0 .+= H.weights[i]*H.stumps[i](xvalid)
    push!(err_valid,sum(sign.(err_valid0) .!= yvalid) / length(yvalid))
end

plot(err)
plot!(err_test)
plot!(err_valid)
xlabel!("Itérations")
ylabel!("Erreur en %")

plot(epsilon)
xlabel!("Itérations")
ylabel!("Epsilon")
ylims!(minimum(epsilon), 0.5)

println("borne d'erreur théorique : ",exp(-2*sum((1/2 .- epsilon).^2))*100,"%")

i = argmin(err_valid)
println(i)
println(err_test[i])
println(minimum(err_valid))


