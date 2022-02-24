import numpy as np
import math
from activations import *

class   MLP():

    def __init__(self, nb_layers, layers_sizes, activations):
        """
        Builds a multilayer-perceptron model matching the given parameters.

        Biases are always considered as the first neuron of a layer, with an
        output fixed to 1, and independant weights data structure.

        Parameters:
        - nb_layers : integer, number of layers in the model (including I/O)
        - layers_sizes : integers array, number of neurons for each layer.
        - activations : strings array, activations functions for each layer.
        """
        # Parameters coherence check (to be replaced with error messages)
        assert(nb_layers == len(layers_sizes))
        assert(nb_layers == len(activations))

        # Parameters loading
        self.nb_layers = nb_layers
        self.layers_sizes = layers_sizes
        self.activations = activations

        # Creation of layers array
        self.layers = []
        for i in range(self.nb_layers):
            self.layers.append(np.zeros(self.layers_sizes[i]))

        # Creation of weights matrixes
        self.weights = []
        for i in range(self.nb_layers - 1):
            matrix = np.random.rand(self.layers_sizes[i], self.layers_sizes[i + 1])
            self.weights.append(matrix)

        # Creation of biases weights array
        self.biases_weights = []
        for i in range(self.nb_layers - 1):
            self.biases_weights.append(np.random.rand(self.layers_sizes[i + 1]))

    def activate_layer(self, i):
        """
            Applies activation function on each neuron in the i layer.
        """
        activation_functions =  {
                                    "softmax": softmax,
                                    "sigmoid": sigmoid,
                                    "relu": ReLU
                                }

        function = activation_functions[self.activations[i].lower()]
        for j, f in enumerate(self.layers[i]):
            self.layers[i][j] = function(f)


    def print_layers(self):
        for layer in self.layers:
            print(layer)


    def feedforward(self, input_data):
        """
        FeedForwards the created model with provided input data.
        Parameters:
            - input_data : array, containing input layer data.
        """
        # To be replaced with error message
        assert(len(input_data) == self.layers_sizes[0])

        # Load input data into the input layer
        for i in range(self.layers_sizes[0]):
            self.layers[0][i] = input_data[i]

        # Layers execution loop
        for i in range(self.nb_layers - 1):
            if i > 0:
                self.activate_layer(i)
            for j in range(self.layers_sizes[i + 1]):

                # Compute weighted sum over all inputs
                w_sum = 0.0
                for k in range(self.layers_sizes[i]):
                    w_sum += self.layers[i][k] * self.weights[i][k][j]

                # Add bias weight
                w_sum += self.biases_weights[i][j]

                # Place weighted sum in the next layer's corresponding neuron
                self.layers[i + 1][j] = w_sum

        # Activate output layer
        return softmax(self.layers[self.nb_layers - 1])


    def compute_cost(self, diagnosis):
        """
            This function computes the model's cost by summing up error on
            each neuron in respect with output_target.
        """
        output_target = [1.0, 0.0] if diagnosis == 'M' else [0.0, 1.0]
        cost = 0.0
        output = self.nb_layers - 1
        length = self.layers_sizes[self.nb_layers - 1]

        for i in range(length):
            cost += math.pow(self.layers[output][i] - output_target[i], 2)
        return cost


    def apply_changes(self):
        for index, matrix in enumerate(self.changes):
            for x, column in enumerate(matrix):
                for y, elem in enumerate(column):
                    nudge = self.changes[index][x][y] / self.nb_rows * self.learning_rate
                    self.weights[index][x][y] -= nudge


    def compute_gradient(self, layer, i, j):
        a = self.layers[layer][i]
        dzw = self.layers[layer - 1][j]
        daz = a * (1.0 - a);
        dca = 2.0 * (a - self.target[i]);
        pre_ret = dzw * daz;
        if layer > 1:
            self.back_target[i] += (pre_ret * self.weights[layer - 1][j][i])
        return pre_ret * dca


    def make_target(self, layer, diagnosis):
        layer_size = self.layers_sizes[layer]

        if layer == self.nb_layers - 1:
            self.target = np.zeros(max(self.layers_sizes[1:len(self.layers_sizes) - 1]))
            self.target[(0 if diagnosis == 'M' else 1)] = 1.0
        else:
            for i in range(layer_size):
                self.target[i] -= self.back_target[i] / layer_size


    def average_changes(self, diagnosis):
        layer = self.nb_layers - 1

        while layer > 0:
            self.make_target(layer, diagnosis)
            self.back_target = np.zeros(self.layers_sizes[layer - 1])
            for i in range(self.layers_sizes[layer]):
                for j in range(self.layers_sizes[layer - 1]):
                    self.changes[layer - 1][j][i] -= self.compute_gradient(layer, i, j)
            layer -= 1

    def gradient_descent(self, diagnosis, input_data):

        self.feedforward(input_data)

        self.average_changes(diagnosis)

        return self.compute_cost(diagnosis)


    def backpropagation(self, df, learning_rate=0.01, max_epoch=1000000):
        data = df.to_numpy()
        epoch = 0
        self.learning_rate = learning_rate
        self.nb_rows = len(df)
        total_cost = 0
        while epoch < max_epoch:
            self.changes = []
            for i in range(self.nb_layers - 1):
                matrix = np.zeros((self.layers_sizes[i], self.layers_sizes[i + 1]))
                self.changes.append(matrix)

            prev_cost = total_cost
            total_cost = 0
            for row in data:
                total_cost += self.gradient_descent(row[2], row[3:])

            print("epoch {0}/{1} - loss: {2} - val_loss: {3}".format(epoch,
                                                                    max_epoch,
                                                                    total_cost,
                                                                    prev_cost - total_cost))
            self.apply_changes()
            epoch += 1



