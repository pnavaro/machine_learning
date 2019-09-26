""" Regularization for Lasso Regression """
struct L1Regularization
    alpha :: Float64
end
    
function(self :: L1Regularization)( w :: Vector{Float64})
    return self.alpha * norm(w)
end

function grad(self :: L1Regularization, w :: Vector{Float64})
    return self.alpha .* sign.(w)
end

"""
    Linear regression model with a regularization factor which does both variable selection 
    and regularization. Model that tries to balance the fit of the model with respect to the training 
    data and the complexity of the model. A large regularization factor with decreases the variance of 
    the model and do para.

    # Parameters:
    
    - `degree`: int
        The degree of the polynomial that the independent variable X will be transformed to.
    - `reg_factor`: float
        The factor that will determine the amount of regularization and feature
        shrinkage. 
    - `n_iterations`: int
        The number of training iterations the algorithm will tune the weights for.
    - `learning_rate`: float
        The step length that will be used when updating the weights.
"""
struct LassoRegression

    degree         :: Int64
    regularization :: L1Regularization
    n_iterations   :: Int64
    learning_rate  :: Float64
    w              :: Vector{Float64}

    function LassoRegression( ; degree, reg_factor, n_features )

        n_iterations  = 3000
        learning_rate = 0.01
        w             = zeros(Float64, n_features) 
 
        new( degree, L1Regularization(reg_factor), n_iterations, 
             learning_rate, w)

    end

    function LassoRegression( ; degree, reg_factor, n_features, 
                              learning_rate, n_iterations)

        w  = zeros(Float64, n_features) 
        new( degree, L1Regularization(reg_factor), n_iterations, 
             learning_rate, w)

    end

end

function fit(self, X, y)
    X = normalize(polynomial_features(X, self.degree))
    # Insert constant ones for bias weights
    X = hcat( ones(eltype(X), size(X)[1]), X)
    self.training_errors = Float64[]
    n_features = size(X)[2]

    # Initialize weights randomly [-1/N, 1/N]
    limit = 1 / sqrt(n_features)
    self.w .= -limit + 2 .* limit .* rand(n_features)

    # Do gradient descent for n_iterations
    for i in 1:self.n_iterations
        y_pred = X * self.w
        # Calculate l2 loss
        mse = mean(0.5 .* (y .- y_pred).^2 + regularization(self.w))
        psuh!(self.training_errors, mse)
        # Gradient of l2 loss w.r.t w
        grad_w  = - (y .- y_pred) * X .+ grad(regularization, self.w)
        # Update the weights
        self.w .-= self.learning_rate .* grad_w
    end

end

function predict(self, X)
    X = normalize(polynomial_features(X, self.degree))
    # Insert constant ones for bias weights
    X = hcat( ones(eltype(X), size(X)[1]), X)
    y_pred = X * self.w
    return y_pred
end
