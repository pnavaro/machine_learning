# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
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

using Pkg
Pkg.add(["NearestNeighbors", "Distances"])

using NearestNeighbors, Distances, Plots


# +
f(x, y) = - sin(x) .+ 0.5 * cos(y) 
nv, np = 2, 1000
data = randn(nv, np)

tree = KDTree(data; leafsize = 10)
nx, ny = 10, 10
xmin, xmax = -1, 1
ymin, ymax = -1, 1
xgrid = LinRange(xmin,xmax, nx)
ygrid = LinRange(ymin,ymax, ny)
mgrid = transpose(hcat(vcat([x for x in xgrid, y in ygrid]...),
                       vcat([y for x in xgrid, y in ygrid]...)))
@show size(grid)

scatter(data[1,:], data[2,:], ms=1)

for x in xgrid
    plot!([x,x], [ymin, ymax], lc = :black, label="")
end
for y in ygrid
    plot!([xmin,xmax], [y, y], lc = :black, label="")
end
plot!()
# -

k = 3
idxs, dists = nn(tree, grid, true);




dists

using Distances
r = colwise(Euclidean(), grid, grid)


