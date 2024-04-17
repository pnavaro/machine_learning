using Statistics
using LinearAlgebra

struct PCA

    n_components :: Int
    mean :: Vector{Float64}
    eigenvectors :: Matrix{Float64}

    function PCA(data, n_components)

        μ = vec(mean(data, dims = 1))
        normalized_data = data .- μ'
        covariance_matrix = cov(normalized_data)
        eigenvalues, eigenvectors = eigen(covariance_matrix)
        sorted_indices = sortperm(eigenvalues, rev = true)

        new( n_components, μ, eigenvectors[:, sorted_indices[1:n_components]])

    end

end

function transform(pca, data)
    normalized_data = data .- pca.mean'
    reduced_data = normalized_data * pca.eigenvectors
    return reduced_data
end

function inverse_transform!(pca, reduced_data)
    reconstructed_data = reduced_data * pca.eigenvectors
    return reconstructed_data
end


data = rand(100,3)

pca = PCA(data, 2)

transform(pca, data)


