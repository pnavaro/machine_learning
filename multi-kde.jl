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

using MultiKDE
using Distributions, Random, Plots

# 1d simulation
bws = [0.05 0.1 0.5]
d = Normal(0, 1)
observations = rand(d, 50)
granularity_1d = 100
x = Vector(LinRange(minimum(observations), maximum(observations), granularity_1d))
ys = []
for bw in bws
    kde = KDEUniv(ContinuousDim(), bw, observations, MultiKDE.gaussian)
    y = [MultiKDE.pdf(kde, _x, keep_all=false) for _x in x]
    push!(ys, y)
end

# 1d plot
highest = maximum([maximum(y) for y in ys])
plot(x, ys, label=bws, fmt=:svg)
plot!(observations, [highest+0.05 for _ in 1:length(ys)], seriestype=:scatter, label="observations", size=(900, 450), legend=:outertopright)

# 2d simulation
dims = [ContinuousDim(), ContinuousDim()]
bws = [[0.3, 0.3], [0.5, 0.5], [1, 1]]
mn = MvNormal([0, 0], [1, 1])
observations = rand(mn, 50)
observations = [observations[:, i] for i in 1:size(observations, 2)]
observations_x1 = [_obs[1] for _obs in observations]
observations_x2 = [_obs[2] for _obs in observations]
granularity_2d = 100
x1_range = LinRange(minimum(observations_x1), maximum(observations_x1), granularity_2d)
x2_range = LinRange(minimum(observations_x2), maximum(observations_x2), granularity_2d)
x_grid = [[_x1, _x2] for _x1 in x1_range for _x2 in x2_range]
y_grid = []
for bw in bws
    kde = KDEMulti(dims, bw, observations)
    y = [MultiKDE.pdf(kde, _x) for _x in x_grid]
    push!(y_grid, y)
end

# 2d plot
highest = maximum([maximum(y) for y in y_grid])
plot([_x[1] for _x in x_grid], [_x[2] for _x in x_grid], y_grid, label=[bw[1] for bw in bws][:, :]', size=(900, 450), legend=:outertopright)
plot!(observations_x1, observations_x2, [highest for _ in 1:length(observations)], seriestype=:scatter, label="observations")
