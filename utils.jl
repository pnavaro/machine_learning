#
# https://github.com/eriklindernoren/ML-From-Scratch
#

import Combinatorics: with_replacement_combinations
using Random, LinearAlgebra

""" Random shuffle of the samples in X and y """
function shuffle_data(X, y, seed)
    rng = MersenneTwister(seed)
    idx = randperm(rng, size(X)[1])
    return X[idx,:], y[idx]
end

""" Random shuffle of the samples in X and y """
function shuffle_data(X, y)
    idx = randperm(size(X)[1])
    return X[idx,:], y[idx]
end

 """ Normalize the dataset X """
function normalize(X)
    l2 = sqrt.(sum(X.^2, dims=1))
    l2 .+= l2 .== 0
    return X / np.expand_dims(l2, axis)
end

function polynomial_features(X, degree)

    n_samples, n_features = size(X)

    function index_combinations()
        combs = [with_replacement_combinations(1:n_features, i) for i in 0:degree+1]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    end
    
    combinations = index_combinations()
    n_output_features = length(combinations)
    X_new = zeros(eltype(X), (n_samples, n_output_features))
    
    for (i, index_combs) in enumerate(combinations)
        X_new[:, i] = prod(X[:, index_combs], dims=2)
    end

    return X_new
end


""" Returns the mean squared error between y_true and y_pred """
function mean_squared_error(y_true, y_pred)
    mse = mean((y_true .- y_pred) .^ 2)
    return mse
end


""" Split the data into train and test sets """
function train_test_split(X :: Array{Float64,2}, y :: Vector{Float64}; 
                          test_size=0.5, shuffle=true, seed=nothing)
    if shuffle
        X, y = shuffle_data(X, y, seed)
    end
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = length(y) - (length(y) ÷ trunc(Int64,1 / test_size)) :: Int64
    X_train, X_test = X[1:split_i,:], X[split_i:end,:]
    y_train, y_test = y[1:split_i], y[split_i:end]

    return X_train, X_test, y_train, y_test
end
