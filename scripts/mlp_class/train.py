from MLP import MLP
from sys import argv, stderr
import numpy as np
import pandas as pd


def main():

    assert(len(argv) == 2)

    mlp = MLP(4, [30, 10, 4, 2], [None, 'sigmoid', 'sigmoid', None]) # explain nones

    df = pd.read_csv(argv[1])

    mlp.backpropagation(df, learning_rate=0.01, max_epoch=1000000)



if __name__ == "__main__":
    main()
