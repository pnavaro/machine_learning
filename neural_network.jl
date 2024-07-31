"""

ref : https://twitter.com/Sumanth_077/status/1786040157109661875

Below is the simple Neural Network consists of 2 layers:

- Hidden Layer
- Output Layer

First Initialize the size of layers along with the weights & biases.

And also define the sigmoid activation function & it's derivative
which is really key to introduce non-linearity.

Forward Pass:

Here the input data is passed through the neural network to obtain the predicted output.

In forward pass, First calculate the output of the hidden layer.

hidden_output = X•W1 + b1

Then apply the sigmoid activation to the output.

output = sigmoid( (X•W1) + b1)

Backward Pass:

First compute the gradients of the output layer.

Loss = (y - output)

Gradient of Loss = (y - output) * sigmoid_derivative(output)

Now calculate d_W2 which is gradient of the loss function with respect to W2.

d_W2 = hidden_output.T • Gradient of Loss

Similarly calculate d_W1, d_b2 & d_b1

dW1: Gradient of the loss function wrt W1

d_b2: Gradient of the loss function wrt b2(bias of neuron in output layer)

d_b1: Gradient of the loss function wrt b1(bias of neuron in hidden layer)

Now Update the Weights:

Here learning rate is the hyper parameter!

A low learning rate can cause the model getting caught in local
optima, while the high learning rate can cause the model to overshoot
the general solution

W1 += learning_rate * d_W1
b1 += learning_rate * d_b1

Now a method to train the neural network using both the forward and backward passes.

The function will run for specified no of epochs, calculating:

1. The Forward Pass
2. Backward Pass
3. Updating the Weights

Finally the Predict Function

Now to predict on any new data all we need to do is a single Forward
Pass through the Network:

"""

using Random
import Statistics: mean

struct NeuralNetwork

    input_size :: Int
    hidden_size :: Int
    output_size :: Int

    W1 :: Matrix{Float64}
    b1 :: Vector{Float64}
    W2 :: Matrix{Float64}
    b2 :: Vector{Float64}

    hidden_output :: Matrix{Float64}
    output :: Matrix{Float64}

	function NeuralNetwork( X, hidden_size, y)

        input_size = size(X, 2) 
        output_size = size(y, 1)

        W1 = randn(input_size, hidden_size)
        b1 = zeros(hidden_size)
        W2 = randn(hidden_size, output_size)
        b2 = zeros(output_size)

        hidden_output = X * W1 .+ b1'
        output = hidden_output * W2 .+ b2'

        new(input_size, hidden_size, output_size, W1, b1, W2, b2, hidden_output, output)

    end

end
		
sigmoid(x) = 1 / ( 1 + exp(-x))

sigmoid_derivative(x) = x * (1 - x)

function forward( nn :: NeuralNetwork, X)

    nn.hidden_output .= sigmoid.(X * nn.W1 .+ nn.b1')
    nn.output .= sigmoid.(nn.hidden_output * nn.W2 .+ nn.b2')
    return nn.output

end

function backward( nn :: NeuralNetwork, X, y, learning_rate)

    d_output = (y' .- nn.output) .* sigmoid_derivative.(nn.output)
    d_W2 = nn.hidden_output' * d_output
    d_b2 = vec(sum(d_output, dims=1))

    d_hidden = d_output * nn.W2' .* sigmoid_derivative.(nn.hidden_output)
    d_W1 = X' * d_hidden
    d_b1 = vec(sum(d_hidden, dims=1))

    nn.W2 .+= learning_rate .* d_W2
    nn.b2 .+= learning_rate .* d_b2
    nn.W1 .+= learning_rate .* d_W1
    nn.b1 .+= learning_rate .* d_b1

end

function train!(nn :: NeuralNetwork, X, y, epochs, learning_rate)

    for epoch in 1:epochs

         output = forward(nn, X)
         backward(nn, X, y, learning_rate)
         loss = mean((y' .- output).^2)

    end

end

predict(nn :: NeuralNetwork, X) = forward(nn, X)


X = randn(10, 3)
y = collect(1:5)
learning_rate = 0.01
epochs = 100

nn = NeuralNetwork(X, 4, y)

print(forward(nn, X))
backward(nn, X, y, learning_rate)
train!(nn, X, y, epochs, learning_rate)
print(forward(nn, X))
