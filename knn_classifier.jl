import CluGen: clugen
using Plots
import StatsBase: countmap


struct KNNClassifier

    k :: Int
    X_train :: Matrix{Float64}
    y_train :: Vector{Int}

    KNNClassifier(k, X, y) = new( k, X, y)

end

euclidean_distance(x1, x2) = sqrt(sum(x1 .- x2).^2)

predict(knn, X) = [predict_single(knn, x) for x in X]

function predict_single(knn, x)

	# Calculate distances between x and all examples in the training set
    distances = [euclidean_distance(x, x_train) for x_train in eachrow(knn.X_train)]

	# Get the indices of the k-nearest neighbors
	k_indices = partialsortperm(distances, 1:knn.k)

	# Get the labels of the k-nearest neighbors
	k_nearest_labels = view(knn.y_train, k_indices)

	# Return the most common class label among the k-nearest neighbors
    most_common = argmax(countmap(k_nearest_labels))

	return most_common

end


# Generate sample data
dimension = 2
n_samples = 400
n_components = 3

options = ([1, 0], pi / 8, [20, 10], 10, 1, 1.5)

o = clugen(dimension, n_components, n_samples, options...)

X = o.points
y_true = o.clusters

scatter(X[:,1], X[:,2], group = y_true)

model = KNNClassifier(5, X, y_true)

y_pred = predict(model, X)

scatter(X[:,1], X[:,2], group = y_true)
