import numpy as np

class   MLP():
    def __init__(self, nb_layers, layers_sizes, activations)
    """
        Builds a multilayer-perceptron model matching the given parameters.

        Biases are always considered as the first neuron of a layer, with an
        output fixed to 1, and independant weights data structure.

        Parameters:
        - nb_layers : integer, number of layers in the model (including I/O)
        - layers_sizes : integers array, number of neurons for each layer.
        - activations : strings array, activations functions for each layer.
    """
        # Parameters loading
        self.nb_layers = nb_layers
        self.layers_sizes = layers_sizes
        self.activations = activations

        # Parameters coherence check (to be replaced with error messages)
        assert(nb_layers == len(layers_size))
        assert(nb_layers == len(activations))

        # Creation of layers array
        self.layers = []
        for i in range(nb_layers)
            self.layers.append(np.zeros(layers_sizes[i]))

        # Creation of weights matrixes
        self.weights = []
        for i in range(nb_layers - 1):
            matrix = np.random.rand(layers_sizes[i], layers_sizes[i + 1])
            self.weights.append(matrix)

        # Creation of biases weights array
        self.biases_weights = []
        for i in range(nb_layers - 1):
            self.biases_weights.append(np.random.rand(layers_sizes[i + 1]))
