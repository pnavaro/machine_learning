# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Julia 1.6.1
#     language: julia
#     name: julia-1.6
# ---

using KernelDensity
using Distributions
using Interpolations
using Plots

# +
import KernelDensity: kernel_dist

kernel_dist(::Type{Epanechnikov},w::Real) = Epanechnikov(0.0,w)

data =  [0.952,0.854,0.414,0.328,0.564,0.196,0.096,0.366,0.902,0.804]

kd   = kde(data, range(-5.0, 5.0,length=2048); kernel=Epanechnikov, bandwidth=1.0*sqrt(5))

dist = InterpKDE(kd, BSpline(Linear()))
# -

scatter(data)


