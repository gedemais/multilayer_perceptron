from MLP import MLP
from sys import argv, stderr
import numpy as np
import pandas as pd

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


def main():

    assert(len(argv) == 2)

    df = pd.read_csv(argv[1])

    mlp.backpropagation(df)


if __name__ == "__main__":
    main()
