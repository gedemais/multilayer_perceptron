from MLP import MLP
from sys import argv, stderr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

usage = "python3 train.py [training_dataset.csv] [matrix_folder]\n"


def train(df_train, w_seed, b_seed):
    """
    This function exports network's weights and biases as a seed file, then
    trains the neural network, before to export final weights and biases as
    files in the matrixes/ folder.
    Parameters :
    - df_train : pandas DataFrame, containing training data.
    - w_seed : 2D matrixes array, containing weights seed.
    - b_seed : 2D matrixes array, containing biases weights seed.
    """
    # Network initialisation
    mlp = MLP(4, [30, 8, 4, 2], [None, 'sigmoid', 'sigmoid', 'softmax'], w_seed, b_seed)

    # Save seeds again (in case we are training a newly randomly weighted network)
    np.save("matrixes/w_seed", mlp.weights)
    np.save("matrixes/b_seed", mlp.biases_weights)

    # Gradient descent training
    mlp.backpropagation(df_train, 0.05, 2000, 110)

    # Loss evolution plot in figs.png with matplotlib
    plt.plot(mlp.cost_data)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("figs" + ".png")


def main():

    if len(argv) != 3:
        stderr.write(usage)
        return 1

    # Parameters loading
    df_train = pd.read_csv(argv[1])
    mx_path = argv[2]

    # Seed loading
    try:
        w_seed = np.load(mx_path + '/' + 'w_seed.npy', allow_pickle=True) # Initial weights
        b_seed = np.load(mx_path + '/' + 'b_seed.npy', allow_pickle=True) # Initial biases
    except FileNotFoundError:
        stderr.write("{} folder not found.\n".format(mx_path))
        return 1
    except PermissionError:
        stderr.write("You don't have required permissions to read files in {}.\n".format(mx_path))
        return 1

    train(df_train, w_seed, b_seed)
    return 0

if __name__ == "__main__":
    main()
