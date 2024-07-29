import numpy as np

"""

ref : https://twitter.com/Sumanth_077/status/1786040157109661875

Below is the simple Neural Network consists of 2 layers:

- Hidden Layer
- Output Layer

First Initialize the size of layers along with the weights & biases.

And also define the sigmoid activation function & it's derivative which is really key to introduce non-linearity.

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

A low learning rate can cause the model getting caught in local optima, while the high learning rate can cause the model to overshoot the general solution

W1 += learning_rate * d_W1
b1 += learning_rate * d_b1

Now a method to train the neural network using both the forward and backward passes.

The function will run for specified no of epochs, calculating:

1. The Forward Pass
2. Backward Pass
3. Updating the Weights

Finally the Predict Function

Now to predict on any new data all we need to do is a single Forward Pass through the Network:

"""


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden_output = self.sigmoid(np.dot(X, self.W1) + self.b1)
        self.output = self.sigmoid(np.dot(self.hidden_output, self.W2) + self.b2)
        return self.output

    def backward(self, X, y, learning_rate):
        d_output = (y - self.output) * self.sigmoid_derivative(self.output)
        d_W2 = np.dot(self.hidden_output.T, d_output)
        d_b2 = np.sum(d_output, axis=0, keepdims=True)

        d_hidden = np.dot(d_output, self.W2.T) * self.sigmoid_derivative(
            self.hidden_output
        )
        d_W1 = np.dot(X.T, d_hidden)
        d_b1 = np.sum(d_hidden, axis=0, keepdims=True)

        self.W2 += learning_rate * d_W2
        self.b2 += learning_rate * d_b2
        self.W1 += learning_rate * d_W1
        self.b1 += learning_rate * d_b1

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)
            loss = np.mean((y - output) ** 2)

    def predict(self, X):
        return self.forward(X)


nn = NeuralNetwork(3, 4, 5)

X = np.random.randn(10, 3)
y = np.arange(5)
learning_rate = 0.01
epochs = 100

print(nn.forward(X))
nn.train(X, y, epochs, learning_rate)
print(nn.predict(X))
