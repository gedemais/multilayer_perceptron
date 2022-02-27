from MLP import MLP
from sys import argv, stderr
import numpy as np
import pandas as pd
from error_functions import *

usage = "python3 evaluate.py [evaluation_dataset.csv] [matrix_folder]\n"

def evaluate(df, w_seed, b_seed):
    """
    This function will evaluate the precision of a model configuration by
    feedforwarding each row in the evaluation dataset in the network, and
    counting the number of right and wrong answers it delivered.
    This will give us a precision percentage, useful to compare configurations.
    It also computes the cross entropy error of the network over evaluation dataset.
    Parameters:
    - df : pandas DataFrame, containing evaluation data.
    - w_seed : 2D matrixes array, containing weights seed.
    - b_seed : 2D matrixes array, containing biases weights seed.
    """

    # Network initialisation
    mlp = MLP(4, [30, 8, 4, 2], [None, 'sigmoid', 'sigmoid', 'softmax'], w_seed, b_seed)

    # Data conversion
    data = df.to_numpy()

    # Variables holding results
    rights = 0
    wrongs = 0
    feecb = 0.0
    # Iterate through evaluation dataset to test answers
    for index, row in enumerate(data):

        mlp.feedforward(row[3:])

        output = mlp.layers[mlp.nb_layers - 1]

        feecb += cross_entropy_error(output[0], 0.0 if row[2] == 'B' else 1.0)
        feecb += cross_entropy_error(output[1], 1.0 if row[2] == 'B' else 0.0)

        # If the answer is benin
        if row[2] == 'B':
            if output[0] >= output[1]:
                wrongs += 1
            else:
                rights += 1
        # If the answer is malin
        else:
            if output[1] >= output[0]:
                wrongs += 1
            else:
                rights += 1

    return wrongs, rights, feecb * -1.0 / len(data)


def main():

    if len(argv) != 3:
        stderr.write(usage)
        return 1

    # Parameters loading
    df = pd.read_csv(argv[1])
    mx_path = argv[2]

    # Seed loading
    try:
        w_seed = np.load(mx_path + '/' + 'weights.npy', allow_pickle=True)
        b_seed = np.load(mx_path + '/' + 'biases.npy', allow_pickle=True)
    except FileNotFoundError:
        stderr.write("{} / {} : No such file or directory.\n".format(mx_path + '/' + 'weights.npy', mx_path +  '/' + 'biases.npy'))
        exit(1)
    except PermissionError:
        stderr.write("You don't have required permissions to read files in matrix_path.\n")
        exit(1)

    wrongs, rights, feecb = evaluate(df, w_seed, b_seed)

    # Output print
    output = "{0} incorrect guesses\n{1} correct guesses\nprecision : {2}\nFEECB loss : {3}"
    print(output.format(wrongs, rights, (rights / (wrongs + rights)), feecb))

    return 0

if __name__ == "__main__":
    main()
