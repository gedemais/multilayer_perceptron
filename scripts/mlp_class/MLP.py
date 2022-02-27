import numpy as np
import math
from sys import argv, stderr
from activations import *
from error_functions import *
from time import sleep

class   MLP():

    def __init__(self, nb_layers, layers_sizes, activations, w_seed=None, b_seed=None):
        """
        Builds a multilayer-perceptron model matching the given parameters.

        Biases are always considered as the first neuron of a layer, with an
        output fixed to 1, and independant weights data structure.

        Parameters:
        - nb_layers : integer, number of layers in the model (including I/O)

        - layers_sizes : integers array, number of neurons for each layer.

        - activations : strings array, activations functions for each layer.

        - w_seed : 2D matrixes array, neurons weights loaded from a seed file.
            If w_seed is equal to None, neurons weights will be set randomly.

        - b_seed : vectors array containing biases weights loaded from a seed file.
            If b_seed is equal to None, neurons weights will be set randomly.
        """
        # Parameters coherence check
        if nb_layers != len(layers_sizes) or nb_layers != len(activations):
            stderr.write("Incoherent dimensions for mlp model.\n")
            exit(1)

        # Parameters loading
        self.nb_layers = nb_layers
        self.layers_sizes = layers_sizes
        self.activations = activations

        # Creation of an array for layers (data-structure holding neurons values)
        self.layers = []
        for i in range(self.nb_layers):
            self.layers.append(np.zeros(self.layers_sizes[i]))

        # Creation of weights matrixes
        if w_seed == None:
            self.weights = []
            for i in range(self.nb_layers - 1):
                matrix = np.random.rand(self.layers_sizes[i], self.layers_sizes[i + 1])
                self.weights.append(matrix)
        else:
            self.weights = w_seed

        # Creation of biases weights array
        if b_seed == None:
            self.biases_weights = []
            for i in range(self.nb_layers - 1):
                self.biases_weights.append(np.random.rand(self.layers_sizes[i + 1]))
        else:
            self.biases_weights = b_seed


    def activate_layer(self, i):
        """
            Applies activation function on each neuron of a layer.
            Parameter :
            - i : integer, index of the layer to activate.

        """
        activation_functions =  {
                                    "softmax": softmax,
                                    "sigmoid": sigmoid,
                                    "relu": ReLU
                                }

        function = activation_functions[self.activations[i].lower()]
        for j, f in enumerate(self.layers[i]):
            self.layers[i][j] = function(f)


    def feedforward(self, input_data):
        """
        Feedforwards the model with user provided input data.
        Parameters:
            - input_data : array, containing input layer data.
        """
        # To be replaced with error message
        if len(input_data) != self.layers_sizes[0]:
            s = 'Invalid data length to feedforward network :\n{} elements\n'
            stderr.write(s.format(len(input_data)))
            return 1

        # Load input data into the input layer
        for i in range(self.layers_sizes[0]):
            self.layers[0][i] = input_data[i]

        # Layers execution loop
        for i in range(self.nb_layers - 1):

            if i > 0: # Activate layer if not first (no activation on first layer)
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
        output = self.nb_layers - 1
        self.layers[output] = softmax(self.layers[output])


    def compute_loss(self, diagnosis):
        """
            This function computes the model's cost by summing up error on
            each neuron in respect with output_target, using the cross entropy
            error formulae specified in 42's subject.
            Parameter :
            - diagnosis : string containing a single char, with two possible
            values : 'M' for malin and "B" for benin.
        """
        output_target = [1.0, 0.0] if diagnosis == 'M' else [0.0, 1.0]

        output = self.nb_layers - 1
        length = self.layers_sizes[output]

        cost = 0.0
        for i in range(length):
            p = self.layers[output][i]
            y = output_target[i]
            cost += cross_entropy_error(p, y)

        return cost * -1.0 / length


    def compute_gradient(self, layer, i, j):
        """
            Computes derivatives to get changes to apply on layer.
            Parameters:
            - layer : index of the layer to compute the gradient.
            - i : previous layer neuron index
            - j : current layer neuron index
        """

        derivatives =   {
                            "softmax": dsoftmax,
                            "sigmoid": dsigmoid,
                            "relu": dReLU
                        }

        # Activation of current neuron
        a = self.layers[layer][i]
        # Activation of previous layer neuron
        dzw = self.layers[layer - 1][j]
        dzb = 1.0

        # Softmax takes a vector as parameter, so it needs its own condition.
        if self.activations[layer] == "softmax":
            daz = dsoftmax(self.layers[layer])[i]
        else:
            daz = derivatives[self.activations[layer]](a)

        # Compute of error according to target
        dca = cross_entropy_error_derivative(a, self.target[i])

        # Compute final change factors for weight and bias linked to i neuron
        weight = dzw * daz * dca
        bias = daz * dca

        # Making weights back target for previous (next) layer
        if layer > 1:
            self.back_target_w[i] += (weight * self.weights[layer - 1][j][i])

        # Adding changes for biases (independant from back layer)
        self.biases_changes[layer - 1][i] += (bias * self.biases_weights[layer - 1][i])

        return weight


    def make_target(self, layer, diagnosis):
        """
            Builds a target values array for a specific layer, from input data.
            Parameters :
            - layer : Index of the layer for which we need a target.
            - diagnosis : Final output that network should deliver.
        """
        layer_size = self.layers_sizes[layer]

        if layer == self.nb_layers - 1:
            self.target = np.zeros(max(self.layers_sizes[1:len(self.layers_sizes) - 1]))
            self.target[(0 if diagnosis == 'M' else 1)] = 1.0
        else:
            for i in range(layer_size):
                self.target[i] -= self.back_target_w[i] / layer_size


    def apply_changes(self):
        """
            Apply changes computed with descent gradient algorithm on weights
            and biases of the network.
        """
        for index, matrix in enumerate(self.changes):
            for x, column in enumerate(matrix):
                for y, elem in enumerate(column):
                    w_nudge = self.changes[index][x][y] / self.nb_rows * self.learning_rate
                    self.weights[index][x][y] -= w_nudge

                    if x == 0:
                        b_nudge = self.biases_changes[index][y] / self.nb_rows * self.learning_rate
                        self.biases_weights[index][y] -= b_nudge


    def average_changes(self, diagnosis):
        """
            Main gradient descent loop, used to iterate backward through
            each layer of the network.
        """
        layer = self.nb_layers - 1

        while layer > 0:
            self.make_target(layer, diagnosis)
            self.back_target_w = np.zeros(self.layers_sizes[layer - 1])
            for i in range(self.layers_sizes[layer]):
                for j in range(self.layers_sizes[layer - 1]):
                    self.changes[layer - 1][j][i] += self.compute_gradient(layer, i, j)
            layer -= 1


    def gradient_descent(self, diagnosis, input_data):

        self.feedforward(input_data)

        self.average_changes(diagnosis)

        return self.compute_loss(diagnosis)


    def backpropagation(self, df, learning_rate=0.05, max_epoch=1000000, early_stop=10000):
        """
            Training function using backpropagation algorithm.
            Parameters:
            - df : pandas DataFrame, containing the training data.
            - learning_rate : float (0-1), Learning speed ratio, useful to tweek during learning phase.
            - max_epoch : integer, Maximum number of epochs to do before stopping network's training.
            - early_stop : float, loss threshold value to automatically stop the training.
        """

        data = df.to_numpy() # Transform data to float arrays
        self.cost_data = [] # Initialize visualisation data structure
        self.learning_rate = learning_rate
        self.nb_rows = len(df) # Number of elements in the training datasets

        total_cost = 0
        epoch = 0
        # Main backpropagation loop
        while epoch < max_epoch:
            # Initialize weights and biases changes matrixes to 0.0
            self.changes = []
            self.biases_changes = []
            for i in range(self.nb_layers - 1):
                matrix = np.zeros((self.layers_sizes[i], self.layers_sizes[i + 1]))
                # Weights changes matrixes
                self.changes.append(matrix)
                # Biases changes vectors
                vector = np.zeros(self.layers_sizes[i + 1])
                self.biases_changes.append(vector)

            prev_cost = total_cost # Used to compute loss delta with last epoch
            total_cost = 0
            for index, row in enumerate(data): # Go through dataset and average changes
                total_cost += self.gradient_descent(row[2], row[3:])

            # Print specified in the subject
            put = "epoch {0}/{1} - loss: {2} - val_loss: {3}"
            print(put.format(epoch, max_epoch, total_cost, prev_cost - total_cost))
            # Visualisation data storage
            self.cost_data.append(total_cost)

            # Early stop implementation
            if total_cost <= early_stop:
                # Save of weights and biases
                np.save("matrixes/weights", self.weights)
                np.save("matrixes/biases", self.biases_weights)
                return 

            self.apply_changes()
            epoch += 1
