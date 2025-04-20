import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to the results CSV (updated location)
RESULTS_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'results', 'fedprox_run_history.csv')

def plot_loss_curve():
    df = pd.read_csv(RESULTS_CSV, header=None, names=['round', 'clients', 'avg_loss'])
    plt.figure(figsize=(10, 6))
    plt.plot(df['round'], df['avg_loss'], marker='o', linestyle='-')
    plt.title('Federated Learning: Average Client Final Loss per Round')
    plt.xlabel('Federated Round')
    plt.ylabel('Average Client Final Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(RESULTS_CSV), 'loss_curve.png'))
    plt.show()

if __name__ == '__main__':
    plot_loss_curve()
