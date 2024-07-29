# https://www.geeksforgeeks.org/single-neuron-neural-network-python/
# Julia program to implement a single neuron neural network 

# Class to create a neural 
# network with single neuron 
using Random

struct SingleNeuron

    weights::Vector{Float64}

    function SingleNeuron()

        rng = MersenneTwister(1)

        weights = 2 .* rand(rng, 3) .- 1

        new(weights)

    end

end

# derivative of tanh function. 
# Needed to calculate the gradients. 
tanh_derivative(x) = 1.0 - tanh(x)^2

# forward propagation 
forward_propagation(model :: SingleNeuron, inputs) = tanh.(inputs * model.weights)

# training the neural network. 
function train!(model :: SingleNeuron, train_inputs, train_outputs, num_train_iterations)

    # Number of iterations we want to perform for this set of input. 
    for iteration = 1:num_train_iterations

        output = forward_propagation(model, train_inputs)

        # Calculate the error in the output. 
        error = train_outputs - output

        # multiply the error by input and then 
        # by gradient of tanh funtion to calculate 
        # the adjustment needs to be made in weights 
        adjustment = train_inputs' * (error .* tanh_derivative.(output))

        # Adjust the weight matrix 
        model.weights .+= adjustment

    end

end

# Driver Code 

nn = SingleNeuron()

println("Random weights at the start of training")
println(nn.weights)

train_inputs = [0. 0 1; 1 1 1; 1 0 1; 0 1 1]
train_outputs = [0., 1, 1, 0]

train!(nn, train_inputs, train_outputs, 10000)

println("New weights after training")
println(nn.weights)

# Test the neural network with a new situation. 
println("Testing network on new examples ->")
println(forward_propagation(nn, [1, 0, 0]'))

