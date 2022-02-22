from MLP import MLP

mlp = MLP(2, [4, 2], ['softmax', 'softmax'])

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
