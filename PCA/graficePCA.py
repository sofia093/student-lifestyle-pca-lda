import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd

''' PCA GRAPHICS '''

# Corelograma
def corelograma(matrix=None, dec=2, title='CORELOGRAMA', valmin=-1, valmax=1):
    """Genereaza o corelograma bazata pe matricea de input"""
    plt.figure(title, figsize=(12, 8))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')

    sb.heatmap(
        data=np.round(matrix, dec),
        vmin=valmin,
        vmax=valmax,
        cmap='bwr',
        annot=True,
        annot_kws={"fontsize": 8},
        linewidths=0.5,
        linecolor='black'
    )

    plt.tight_layout()


# Intensitatea legaturilor
def linkIntesity(matrix=None, dec=2, title='INTENSITATEA LEGATURII'):
    """Genereaza o harta de caldura pentru intensitatea legaturilor"""
    plt.figure(title, figsize=(15, 11))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')

    sb.heatmap(
        data=np.round(matrix, dec),
        cmap='Oranges',
        annot=True,
        annot_kws={"fontsize": 8},
        linewidths=0.5,
        linecolor='black'
    )

    plt.yticks(rotation=0)
    plt.tight_layout()


# Cercul corelatiilor
def cercCorelatie(matrix=None, V1=0, V2=1, dec=1, XLabel=None, YLabel=None, title='CERCUL DE CORELATIE'):
    """Genereaza un cerc al corelațiilor pentru primele două componente"""
    plt.figure(title, figsize=(10, 10))
    plt.title(title, fontsize=16, color='k', verticalalignment='bottom')

    # Cerc
    T = [t for t in np.arange(0, np.pi * 2, 0.01)]
    X = [np.cos(t) for t in T]
    Y = [np.sin(t) for t in T]
    plt.plot(X, Y, linestyle='--', color='blue')

    # Axele X si Y
    plt.axhline(y=0, color='g', linestyle='-')
    plt.axvline(x=0, color='g', linestyle='-')

    # Punctele din matrice
    if isinstance(matrix, pd.DataFrame):
        plt.scatter(x=matrix.iloc[:, V1], y=matrix.iloc[:, V2], c='b')

        for i in range(matrix.shape[0]):
            x = matrix.iloc[i, V1]
            y = matrix.iloc[i, V2]
            offset_x = 0.03 if x > 0 else -0.03
            offset_y = 0.03 if y > 0 else -0.03
            plt.text(
                x + offset_x,
                y + offset_y,
                s=matrix.index.values[i],
                fontsize=10,
                color='darkblue',
                ha='center',
                va='center'
            )

    # Etichete
    if XLabel:
        plt.xlabel(XLabel, fontsize=14, color='k', verticalalignment='top')
    else:
        plt.xlabel(f"C{V1 + 1}", fontsize=14, color='k', verticalalignment='top')

    if YLabel:
        plt.ylabel(YLabel, fontsize=14, color='k', verticalalignment='bottom')
    else:
        plt.ylabel(f"C{V2 + 1}", fontsize=14, color='k', verticalalignment='bottom')

    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)


# Componentele principale
def componentePrincipale(eigenvalues=None, XLabel='Componentele principale', YLabel='Varianta valorilor proprii',
                         title='Varianta explicata de componentele principale'):
    """Genereaza un grafic pentru varianta explicata de componentele principale."""
    plt.figure(title, figsize=(13, 8))
    plt.title(title, fontsize=14, color='k', verticalalignment='bottom')
    plt.xlabel(XLabel, fontsize=14, color='k', verticalalignment='top')
    plt.ylabel(YLabel, fontsize=14, color='k', verticalalignment='bottom')
    components = ['C' + str(j + 1) for j in range(eigenvalues.shape[0])]
    plt.plot(components, eigenvalues, 'bo-')
    plt.axhline(y=1, color='r')  # Linia de referinta (criteriul Kaiser)

def afisareGrafic():
    """Afiseaza graficul curent."""
    plt.show()
def salveazaGrafic(cale):
    """Salveaza graficul curent in calea specificata."""
    plt.savefig(cale, bbox_inches='tight')
    plt.close()
