from MLP import MLP
from sys import argv, stderr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cProfile

def test(couches, layers, acts, df, lr, me, name):
    mlp = MLP(couches, layers, acts) # explain nones
    mlp.backpropagation(df, learning_rate=lr, max_epoch=me)
    plt.plot(mlp.cost_data)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("figs/" + name + ".png")
    return mlp

def main():

    assert(len(argv) == 3)

    df = pd.read_csv(argv[1])
    df_test = pd.read_csv(argv[2])

    mlp = test(4, [30, 8, 4, 2], [None, 'sigmoid', 'sigmoid', "softmax"], df, 0.1, 400, "30, 4, 4, 2")

    data = df_test.to_numpy()
    for row in data:
        print(row[2])
        mlp.feedforward(row[3:])
        print(mlp.layers[mlp.nb_layers - 1])
        print('-' * 30)

if __name__ == "__main__":
    main()
