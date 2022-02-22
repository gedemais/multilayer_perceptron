from MLP import MLP
import numpy as np

mlp = MLP(4, [30, 15, 8, 2], [None, 'sigmoid', 'sigmoid', None]) # explain nones

#print(mlp.nb_layers)
#print('-' * 80)
#print(mlp.layers_sizes)
#print('-' * 80)
#print(mlp.activations)
#print('-' * 80)
#print(mlp.layers)
#print('-' * 80)
#print(mlp.weights)
#print('-' * 80)
#print(mlp.biases_weights)

input_data = np.random.rand(30)

print("input_data = {}".format(input_data))

output = mlp.feedforward(input_data)

print("output = {}".format(output))
