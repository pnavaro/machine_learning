# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Julia 1.10.2
#     language: julia
#     name: julia-1.10
# ---

# ## Generate observations from a logit model

# ## Ordinary Least Square (OLS) estimation

# +
n, k = 1000, 3

X = hcat(ones(n), randn(n, k))

epsilon = randn(n)
trueParams = [0.1, 0.5, -0.3, 0.]
Y = X * trueParams .+ epsilon
# -

function ols_estimator(y,x)
    estimate = inv(x'x)*(x'y)
    return estimate
end
estimates = ols_estimator(Y,X)

# ## Maximum likelihood estimation

# +
function loglike(rho, y, x)
    beta = rho[1:4]
    sigma2 = exp(rho[5])+eps(Float64)
    residual = y - x * beta
    dist = Normal(0, sigma2)
    contributions = logpdf(dist,residual)
    loglikelihood = sum(contributions)
    return -loglikelihood
end

params0 = [.1,.2,.3,.4,.5]
wrap_loglike(rho) = loglike(rho,Y,X)
optimum = optimize(wrap_loglike, params0, BFGS())
mle = optimum.minimizer
mle[5] = exp(mle[5])
mle
# -

# ## Bootstrapping and Non-parametric p-values

function bootstrap(b = 1000)
    samples = zeros(b,5)
    indices = collect(1:n)
    for b = 1:b
        shuffle!(indices)
        x = view(X, indices, :)
        y = view(Y, indices, : )
        wrap_loglike(rho) = loglike(rho, y, x)
        samples[b,:] .= optimize(wrap_loglike, params0, BFGS()).minimizer
    end
    samples[:,5] .= exp.(view(samples,:,5))
end

# +
bootstrap_se = std(samples,dims=1)

null_distribution = samples
pvalues = ones(5)
for i=1:5
    null_distribution[:,i] .= null_distribution[:,i] .- mean(null_distribution[:,i])
end
null_distribution[:,5] .= 1 .+ nullDistribution[:,5]

pvalues = [mean(abs.(mle[i]) .< abs.(null_distribution[:,i])) for i=1:5]

# +
using LinearAlgebra
using Plots
using Random
import Statistics: std, mean
import Distributions: Normal, logpdf
import Optim: BFGS, optimize


n = 1000

rng = Xoshiro(1234)

β = [1, -2, 3, 0, 6, -1, 0]


sigmoid(x) =  1 / (1 + exp(-x))
function logit(n, β)
    k = length(β)
    X = ones(n)
    if k > 1
        X = [X  randn(n,k-1)]
    end
    y = round.(Int, sigmoid.(- X * β) .> rand(n))
    return X, y
end
X, y = logit(1000, [1,2,3])

sigmoid_gradient(x) =  sigmoid(x) * (1 - sigmoid(x))
function fit(X, y, n_iterations=100)
    n_obs , n_features = size(X)
    β = ones(n_features)
    for i in 1:n_iterations
        x = X * β
        y_pred = sigmoid.(x)
        d = Diagonal(sigmoid_gradient.(x))
        newbeta = pinv(X' * d * X) * X' * (d * X * β + y - y_pred)
        if newbeta ≈ β
            return newbeta
        else
            β .= newbeta
        end
    end
end

β = fit(X, y)
predict(β, X) = round.(Int, sigmoid.(X * β))
scatter( X[:,2], X[:, 3], markercolor = predict(β, X), aspect_ratio = :equal)
# -

# ## Forward regression to select variables

# +
function forward_regression(X, y; threshold_in = 0.01)
    included = Int[]
    nvars = size(X, 1)
    while true
        changed=False
        excluded = [i for i 1:nvars if i ∉ included]
        new_pval = fill(Inf, nvars)
        for new_column ∈ excluded
            X̃ = sm.add_constant(X[:,included+[new_column]])
            model = sm.Logit(y, X̃).fit(disp=False)
            new_pval[new_column] = model.pvalues[-1]
        end
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = np.argmin(new_pval)
            included.append(best_feature)
            changed=True
        end
        if not changed:
            return included
        end
    end
end

forward_regression(X, y, 0.05)


# +
def backward_regression(X, y, threshold_out = 0.01):
    included=list(range(np.size(X,1))) # all variables
    while True:
        changed=False
        X̃ = sm.add_constant(X[:,included])
        model = sm.Logit(y, X̃).fit(disp=False)
        # use all coefs except intercept
        pvalues = model.pvalues[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        print(included)
        if worst_pval > threshold_out:
            changed=True
            worst_feature = included[np.argmax(pvalues)]
            included.remove(worst_feature)
        if not changed:
            return included
    

backward_regression(X, y)


# +
def modelcomparison(X, y, samples):
    """
    takes as input argument the sample and a set of competing models. 
    This function returns the best model (i.e., the subset of the relevant variables) 
    and its estimator of the prediction error.
    """
    pass


def modelselection(X, y, direction):
    """
    takes as input argument the sample and the direction (backward or forward). 
    This function returns the best model (i.e., the subset of the relevant variables) 
    and its estimator of the prediction error.
    """
    models = dict(forward = forward_regression, backward = backward_regression)
    
    return models[direction](X, y)
    
    
# -


