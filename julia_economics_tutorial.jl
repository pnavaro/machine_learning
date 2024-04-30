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

# # Julia economics tutorials
#
# https://github.com/setzler/JuliaEconomics/tree/master/Tutorials
#
# - [INTRODUCTORY EXAMPLE: ORDINARY LEAST SQUARES (OLS) ESTIMATION IN JULIA](https://juliaeconomics.com/2014/06/15/introductory-example-ordinary-least-squares/)
# - [MAXIMUM LIKELIHOOD ESTIMATION (MLE) IN JULIA: THE OLS EXAMPLE](https://juliaeconomics.com/2014/06/16/numerical-maximum-likelihood-the-ols-example/)
# - [BOOTSTRAPPING AND NON-PARAMETRIC P-VALUES IN JULIA](https://juliaeconomics.com/2014/06/16/bootstrapping-and-hypothesis-tests-in-julia/)
# - [STEPDOWN P-VALUES FOR MULTIPLE HYPOTHESIS TESTING IN JULIA](https://juliaeconomics.com/2014/06/17/stepdown-p-values-for-multiple-hypothesis-testing-in-julia/)
#
# BRADLEY SETZLER www.BradleySetzler.com

# ## Packages

# +
import Distributions: Normal, logpdf, fit
using LinearAlgebra
import Optim: optimize, NelderMead
using Random
using ProgressMeter
using Statistics
import StatsBase: sample

rng = MersenneTwister(123)
# -

# ## Ordinary Least Square (OLS) estimation

# As a reminder and to clarify notation, the OLS data generating process is,
#
# $$
# Y = X\beta + \epsilon,
# $$
#
# where $Y$ is the $Nx1$ dependent variable, $X$ is the $N\mathrm{x}\tilde{K}$ matrix of independent variables, $\beta$ is the $\tilde{K}\mathrm{x}1$ vector of parameters that we wish to estimate, and $\epsilon$ is the $N\mathrm{x}1$ error satisfying 
#
# $$
# \epsilon \overset{\mathit{i.i.d.}}{\sim} \mathcal{N}\left(0,\sigma^2\right).
# $$
#
# Because we assume that the first column of $X$ is the constant number 1, we will find it useful below to work with$ K \equiv \tilde{K}-1$. The least squares estimator is,
#
# $$
# \hat{\beta} = \left( X^T X \right)^{-1} \left( X^T Y \right).
# $$

# +
n = 1000
k = 3

X = hcat(ones(n), randn(rng, n, k))
β = [0.1, 0.5, -0.3, 0.]
ϵ = randn(rng, n)
Y = X * β .+ ϵ

ols_estimator(y,x) = inv(x'x) * x'y

ols_estimator(Y,X)
# -

# ## Maximum likelihood estimation
#
# Denote the parameter vector by $\rho \equiv [\beta, \sigma^2]$. We will now see how to obtain the MLE estimate $\hat\rho$ of $\rho$. By Bayes rule and independence across individuals $(i)$, the likelihood of $\rho$ satisfies,
#
# $$
# \mathcal{L}\left(\rho|Y,X\right) \propto \prod_{i=1}^N \phi\left( Y_i - X_i\beta, \sigma^2 \right|\rho);
# $$
#
# where $\phi$ is the normal probability distribution function (PDF). $\hat\rho$ is the $\arg\max$ of this expression, and we will show how to find it using a numerical search algorithm in Julia.
#
# This code first collects $\beta$ and $\sigma^2$ from $\rho$, uses $\sigma^2$ to initialize the appropriate normal distribution $d$, then evaluates the normal distribution at each of the residuals, $Y_i-X_i\beta$ (residuals), returning the negative of the sum of the individual contributions to the log-likelihood (contributions).
#
# *Tip: Always remember to use the negative of the likelihood (or log-likelihood) that you wish to maximize, because the optimize command is a minimizer by default. Since the $\arg\max$ is the same when maximizing some function and minimizing the negative of the function, we need the negative sign in order to maximize the likelihood with a minimizer.*
#
# The only confusing part of this function is that $\sigma^2$ is read from $\rho$ as an exponential. This is strictly a unit conversion — it means that $\sigma^2$ was stored in $\rho$ in log-units, so we must exponentiate to return it to levels, that is, $\rho$ is defined as $\rho \equiv [\beta, \log\left(\sigma^2\right)]$. This is a common approach in numerical optimization to solve a practical problem: the numerical optimizer tries out many possible values for $\rho$ in its effort to search for the MLE, and it may naively try out a negative value of $\sigma^2$, which would be a logical contradiction that would crash the code. By storing $\sigma^2$ in $\log$ units, we make it perfectly acceptable for the optimizer to try out a negative value of the parameter, because a parameter that is negative in log-units is non-negative in level-units.
#
# *Tip: Always restrict variances so that the optimizer cannot try negative values. We have used log-units to achieve this goal.*

# +
function loglike(ρ, y, x)
    β = ρ[1:end-1]
    σ² = exp(ρ[end])
    residual = y - x * β
    d = Normal(0, sqrt(σ²))
    contributions = logpdf(d,residual)
    loglikelihood = sum(contributions)
    return -loglikelihood
end

function maximum_likelihood_estimation(x, y)
    β₀ = zeros(size(x,2)+1)
    optimum = optimize(ρ -> loglike(ρ, y, x), β₀, NelderMead())
    mle = optimum.minimizer
    mle[5] = exp(mle[end])
    mle
end

maximum_likelihood_estimation(X, Y) 
# -

# This says to optimize the function `loglike`, starting from the point $\beta_0$, which is chosen somewhat arbitrarily. Numerical search algorithms have to start somewhere, and $\beta_0$ serves as an initial guess of the optimum. Of course, the best possible guess is $\beta$, because the optimizer would have to do much less work, but in practice, we do not know the true parameters so the optimizer will have to do the work. Notice that, at the end, we have to exponentiate the $\sigma^2$ parameter because the optimizer will return it in log-units due to our exponentiation above.

# ## Bootstrapping and Non-parametric p-values

# Now, we will use a random index, which is drawn for each `b` using the sample function, to take a random sample of individuals from the data, feed them into the function using a wrapper, then have the optimizer maximize the wrapper across the parameters. We repeat this process in a loop, so that we obtain the MLE for each subset. The following loop stores the MLE in each row of the matrix samples using 1,000 bootstrap samples of size one-half (M) of the available sample:

# +
function bootstrapping(X, Y, b = 1000)
    n = size(X, 1)
    m = size(X, 2)+1
    β₀ = zeros(m)
    samples = zeros(b,m)
    x = similar(X)
    y = similar(Y)
    batch = collect(1:n)
    
    @showprogress 1 for i = 1:b
        batch .= sample(1:n,n)
        x .= X[batch,:]
        y .= Y[batch,:]
        samples[i,:] .= optimize(ρ -> loglike(ρ, y, x), β₀, NelderMead()).minimizer
        
    end
    samples[:,end] .= exp.(samples[:,end])
    samples
    
end

# +
function null_distribution( samples )
    result =  samples .- mean(samples, dims=1)
    result[:,end] .+= 1
    result
end

function pvalues( mle, null_d)   
    [mean(abs(mle[i]) .< abs.(view(null_d,:,i))) for i=eachindex(mle)]
end
        
samples = bootstrapping(X, Y, 100)
mle = maximum_likelihood_estimation( X, Y)
pvalues(mle, null_distribution(samples))
# -

# Standard errors like these can be used directly for hypothesis testing under parametric assumptions. For example, if we assume an MLE is normally distributed, then we reject the null hypothesis that the parameter is equal to some point if the parameter estimate differs from the point by at least 1.96 standard errors (using that the sample is large). However, we can make fewer assumptions using non-parametric p-values. The following code creates the distribution implied by the null hypothesis that $\beta_0 =0, \beta_1=0, \beta_2=0, \beta_3=0, \sigma^2=1$ by subtracting the mean from each distribution (thus imposing a zero mean) and then adding 1 to the distribution of $\sigma^2$ (thus imposing a mean of one); this is called nullDistribution.

# Suppose that we wish to test to see if the the parameter estimates of $\beta$ are statistically different from zero and if the estimate of $\sigma^2$ is different from one for the OLS parameters defined above. Suppose further that we do not know how to compute analytically the standard errors of the MLE parameter estimates.
#
# We decide to bootstrap by resampling cases in order to estimate the standard errors. This means that we treat the sample of `N` individuals as if it were a population from which we randomly draw `B` samples, each of size `N`. This produces a sample of MLEs of size B, that is, it provides an empirical approximation to the distribution of the MLE. From the empirical approximation, we can compare the full-sample point MLE to the MLE distribution under the null hypothesis.

n = 1000
β = [0.1, 0.5, 0.0001, 0.5, 0.7, 0.0001]
X = hcat(ones(n), randn(rng, n, length(β)-1))
ϵ = randn(rng, n)
Y = X * β .+ ϵ
mle = maximum_likelihood_estimation(X, Y)
samples = bootstrapping(X, Y, 100)
p = pvalues(mle, null_distribution(samples))

# Thus, we reject the null hypotheses for the first, third and fourth parameters only and conclude that $\beta_0 \neq 0, \beta_2 \neq 0, \beta_3 \neq 0$, but find insufficient evidence to reject the null hypotheses that $\beta_1 =0$ and $\sigma^2=1$.

# ## Holm’s Correction to the p-values
#
# Suppose you have a vector of p-values, p, of length K, and a desired maximum family-wise error rate, $\alpha$ (the most common choice is $\alpha=0.05$. These are all of the needed ingredients. Holm’s stepdown procedure is as follows:
#
#  1. If the smallest element of $p$ is greater than $\frac{\alpha}{(K+1)}$, reject no null hypotheses and exit (do not continue to step 2). Else, reject the null hypothesis corresponding to the smallest element of $p$ (continue to step 2).
#  2. If the second-smallest element of p is greater than $\frac{\alpha}{(K+1)-1}$, reject no remaining null hypotheses and exit (do not continue to step 3). Else, reject the null hypothesis corresponding to the second-smallest element of $p$ (continue to step 3).
#  3. If the third-smallest element of p is greater than $\frac{\alpha}{(K+1)-2}$, reject no remaining null hypotheses and exit (do not continue to step 4). Else, reject the null hypothesis corresponding to the third-smallest element of $p$ (continue to step 4).
#  4. And so on.
#  
# We could program this directly as a loop-based function that takes $p,\alpha$ as parameters, but a simpler approach is to compute the p-values equivalent to this procedure for any $\alpha$, which are,
#
# $$
# \tilde{p}_{(k)} =\min\left\{\max_{j:p_{(j)}\leq p_{(k)}} \left\{ (K+1-j)p_{(j)}\right\},1\right\},
# $$
#
# where (k) means the $k^{\mathit{th}}$ smallest element. This expression is equivalent to the algorithm above in the sense that, for any family-wise error rate $\alpha\in (0,1), \tilde{p}_{(k)}\leq\alpha$ if and only if the corresponding hypothesis is rejected by the algorithm above.
#
# The following code computes these p-values. Julia (apparently) lacks a command that would tell me the index of the rank of the p-values, so my loop below does this, including the handling of ties (when some of the p-values are the same):

reverse(1:5) |> collect

# +
function holm(p)
    K = length(p)
    sort_index = -ones(Int, K)
    sorted_p = sort(p)
    sorted_p_adj = sorted_p .* reverse(1:K)
    
    for j=1:K
        num_ties = length(sort_index[(p .== sorted_p[j]) .&& (sort_index .< 0)])
        sort_index[(p.==sorted_p[j]) .&& (sort_index.<0)] .= j:(j-1+num_ties)
    end
    sorted_holm_p = [minimum([maximum(sorted_p_adj[1:k]),1]) for k=1:K] 
    return sorted_holm_p[sort_index]
end

holm(p[1:end-1])
# -


# This is straight-forward except for `sort_index`, which I constructed such that, e.g., if the first element of `sort_index` is 3, then the first element of pvalues is the third smallest. Unfortunately, it arbitrarily breaks ties in favor of the parameter that appears first in the MLE array, so the second entry of sort_index is 1 and the third entry is 2, even though the two corresponding p-values are equal.

# ## Bootstrap Stepdown p-values
#
# Given our work yesterday, it is relatively easy to replace bootstrap marginal p-values with bootstrap stepdown p-values. We begin with samples, as created yesterday. The following code creates tNullDistribution, which is the same as nullDistribution from yesterday except as t-statistics (i.e., divided by standard error).

samples = bootstrapping( X, Y, 100 )
mle = maximum_likelihood_estimation( X, Y)

# +
function stepdown(mle, samples)
    t_mle = mle[1:end-1]
    t_null_distribution = samples[:,1:end-1]
    bootstrap_se = vec(std(t_null_distribution,dims=1))
    p = fill!(similar(t_mle), -1)
    for i = eachindex(t_mle)
        t_mle[i] = abs(t_mle[i]/bootstrap_se[i])
        t_null_distribution[:,i] .= abs.((t_null_distribution[:,i] .- mean(t_null_distribution[:,i]))/bootstrap_se[i])
        p[i] = mean(t_mle[i] .< t_null_distribution[:,i])
    end
    K = length(p)
    sort_index = - ones(Int, K)
    stepdown_p = fill!(similar(p), -1)
    @show sorted_p = sort(p)
    for j=1:K
        num_ties = length(sort_index[(p .== sorted_p[j]) .&& ( sort_index .<0 )])
        sort_index[(p .== sorted_p[j]) .&& (sort_index .< 0)] = j:(j-1+num_ties)
    end
    for k=1:K
        current_index = [sort_index.>=k]
        stepdown_p[sort_index[k]] = mean(maximum(t_null_distribution[:,sort_index.>=k],dims=2) .> t_mle[sort_index[k]])
    end
    return ["single_pvalues"=>p,"stepdown_pvalues"=>stepdown_p,"Holm_pvalues"=>holm(p)]
end

stepdown(mle, samples)
# -


# The only difference between the single p-values and the stepdown p-values is the use of the maximum t-statistic in the comparison to the null distribution, and the maximum is taken over only the parameter estimates whose p-values have not yet been computed. Notice that I used a dictionary in the return so that I could output single, stepdown, and Holm p-values from the stepdown function.

# ## Results
#
# To test out the above corrected p-values, we return to MLE and samples from yesterday. Suppose we want to perform the two-sided tests simultaneously for the null hypotheses \beta_0=0,\beta_1=0,\beta_2=0,\beta_3=0. The results from the above code are,
#
#




# +
function single_pvalues(mle, samples)
    t_mle = mle[1:end-1]
    t_null_distribution = samples[:,1:end-1]
    bootstrap_se = vec(std(t_null_distribution,dims=1))
    p = fill!(similar(t_mle), -1)
    for i = eachindex(t_mle)
        t_mle[i] = abs(t_mle[i]/bootstrap_se[i])
        t_null_distribution[:,i] .= abs.((t_null_distribution[:,i] .- mean(t_null_distribution[:,i]))/bootstrap_se[i])
        p[i] = mean(t_mle[i] .< t_null_distribution[:,i])
    end
    p
end

p = single_pvalues(mle, samples)
# -



# We see that the p-value corresponding to the null hypothesis $\beta_2=0$ is inflated by both the bootstrap stepdown and Holm procedures, and is no longer significant at the 0.10 level.
#
#

 using HypothesisTests

linreg(x, y) = hcat(fill!(similar(x), 1), x) \ y



coefs = ols_estimator(Y, X)
lr_predict( x, coefs) = muladd(@view(x[:, 2:end]), @view(coefs[2:end]), coefs[1])
residuals = Y - lr_predict(X, coefs)
fitted_residuals = fit(Normal, residuals)
kst = HypothesisTests.ApproximateOneSampleKSTest(residuals, fitted_residuals)
pval = pvalue(kst)

# +
using Plots

plot(residuals)
fitted_residuals
# -


