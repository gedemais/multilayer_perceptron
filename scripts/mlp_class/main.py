from MLP import MLP

mlp = MLP(4, [30, 16, 8, 2], ['softmax', 'softmax', 'softmax', 'softmax'])

print(mlp.nb_layers)
print('-' * 80)
print(mlp.layers_sizes)
print('-' * 80)
print(mlp.activations)
print('-' * 80)
print(mlp.layers)
print('-' * 80)
print(mlp.weights)
print('-' * 80)
print(mlp.biases_weights)
