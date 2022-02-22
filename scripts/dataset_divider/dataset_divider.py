import pandas as pd
from sys import argv, stderr


usage = "Usage : python3 dataset_divider.py dataset_path.csv\n"


def divide(df):
    """
    This function splits the dataset in two groups by diagnosis.
    Then it builds a training and an evaluation datasets from theses groups,
    with a ratio of 80% data reserved for training, and 20% remaining for test.
    """
    # Separate rows by diagnosis
    group = df.groupby(df['M'])
    benins = group.get_group('B')
    malins = group.get_group('M')

    training_part = 0.8 # Porportion of rows reserved for training

    # Compute the number of rows to reserve according to training_part
    b_training_features = int(len(benins) * training_part)
    m_training_features = int(len(malins) * training_part)

    # Concatenate first benins and malins in the training dataset
    df_training = pd.concat([benins.iloc[0:b_training_features],
                                malins.iloc[0:m_training_features]])

    # Concatenate last benins and malins in the evaluation dataset
    df_evaluation = pd.concat([benins.iloc[b_training_features:],
        malins.iloc[m_training_features:]])

    # Return shuffled datasets (not shure if necessary, but more realistic)
    return df_training.sample(frac=1), df_evaluation.sample(frac=1)


def main(argv):
    # Parameters check
    if len(argv) != 2:
        stderr.write("Invalid number of arguments.\n" + usage)
        exit(1)

    # CSV dataset parsing
    try:
        df = pd.read_csv(argv[1])
        #df = df.drop(["Index", "First Name", "Last Name", "Birthday", "Best Hand"], 1)
    except:
        print("CSV parsing failed. Abort.")
        exit(1)

    # Dataset division
    df_training, df_evaluation = divide(df)

    # Exports training and evaluation datasets to csv
    df_training.to_csv('training_dataset.csv')
    df_evaluation.to_csv('evaluation_dataset.csv')


if __name__ == "__main__":
    main(argv)
