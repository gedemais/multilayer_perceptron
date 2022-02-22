import pandas as pd
from sys import argv, stderr

usage = "Usage : python3 dataset_divider.py dataset_path.csv\n"

def divide(df):
    """
    This function splits the dataset in two groups following their diagnosis.
    Then it builds a training and an evaluation datasets from theses groups,
    with a ratio of 80% data reserved for training, and 20% remaining for test.
    """
    group = df.groupby(df['M'])
    malins = group.get_group('M')
    benins = group.get_group('B')

    #print(malins)
    #print(benins)
    training_part = 0.8

    b_training_features = int(len(benins) * training_part)
    m_training_features = int(len(malins) * training_part)

    df_training = pd.concat([benins.iloc[0:b_training_features], malins.iloc[0:m_training_features]], axis=1)
    df_evaluate = pd.concat([benins.iloc[b_training_features:], benins.iloc[m_training_features:]], axis=1)
    print(df_training)
    return df_training, df_evaluate

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

    #print(df)
    #print('-' * 80)
    df_training, df_evaluate = divide(df)


if __name__ == "__main__":
    main(argv)