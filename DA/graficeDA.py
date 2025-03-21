import matplotlib.pyplot as plt
import seaborn as sns

# Functie pentru scatter plot cu doua seturi de date
def plot_scatter_groups(x, y, g, labels, x1, y1, g1, labels1, title, lx, ly, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=x, y=y, hue=g)
    sns.scatterplot(x=x1, y=y1, hue=g1, marker='X', s=100, legend=False)
    plt.title(title)
    plt.xlabel(lx)
    plt.ylabel(ly)
    if save_path:
        plt.savefig(save_path)
    plt.close()

# Functie pentru distributia unei variabile pe grupuri
def plot_distribution(z, y, g, title, save_path=None):
    plt.figure(figsize=(10, 6))
    for group in g:
        sns.kdeplot(z[y == group], label=f'Group {group}')
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()
