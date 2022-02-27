import matplotlib.pyplot as plt
import json

def main():
    """
    This script will plot training stats stored in data/loss_stats, in order
    to allow user to visually search for loss sweetpoint (before overfitting,
    but with the highest precision rate reachable). This data was created by
    launching 8 different training periods with different early_stop values
    from 160 to 80, and evaluating theses configurations to plot correlations
    between loss value, errors number, and precision percentage.
    """
    # Loading json data
    with open('data/loss_stats.json', 'r') as f:
        stats = json.load(f)

    # Generating lists from specific fields in stats dict.
    loss_perf = [stat['precision'] for stat in stats]
    loss_errors = [stat['errors'] for stat in stats]

    # Normalization of values in order to get a nice plot.
    loss_perf = [float(i)/max(loss_perf) for i in loss_perf]
    loss_errors = [float(i)/max(loss_errors) for i in loss_errors]

    # Plot
    plt.plot(loss_perf)
    plt.plot(loss_errors)
    plt.show()

if __name__ == "__main__":
    main()
