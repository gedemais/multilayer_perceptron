from MLP import MLP
from sys import argv, stderr
import numpy as np
import pandas as pd


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


def main():

    assert(len(argv) == 2)

    df = pd.read_csv(argv[1])

    mlp = MLP(3, [4, 3, 2], [None, 'sigmoid', None]) # explain nones

    input_data = [0.45, 0.12, 0.87, 0.61]
    input_data = [0.61, 0.87, 0.12, 0.45]
    output = mlp.feedforward(input_data)
    print(output)


if __name__ == "__main__":
    main()
