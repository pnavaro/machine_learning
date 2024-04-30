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
sigmoid_gradient(x) =  sigmoid(x) * (1 - sigmoid(x))

function logit(n, β)
    k = length(β)
    X = ones(n)
    if k > 1
        X = [X  randn(n,k-1)]
    end
    y = round.(Int, sigmoid.(- X * β) .> rand(n))
    return X, y
end

X, y = logit(1000, β)


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
plot(
scatter( X[:,2], X[:, 3], markercolor = predict(β, X), aspect_ratio = :equal),
scatter( X[:,5], X[:, 6], markercolor = predict(β, X), aspect_ratio = :equal))
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
function backward_regression(X, y, threshold_out = 0.01)
    included = collect(1:size(X,1)) # all variables
    while True
        changed=false
        X̃ = sm.add_constant(X[:,included])
        model = sm.Logit(y, X̃).fit(disp=False)
        # use all coefs except intercept
        pvalues = model.pvalues[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        print(included)
        if worst_pval > threshold_out
            changed=True
            worst_feature = included[np.argmax(pvalues)]
            included.remove(worst_feature)
        end
        if not changed
            return included
        end
    end
end

backward_regression(X, y)
# -



