import numpy as np

class PCA:
    def __init__(self, X, regularizare=True):
        """
        Constructorul clasei PCA pentru analiza componentelor principale
        Parametri:
        - X: Matricea de date originale (observatii x variabile)
        - regularizare: Daca True, regularizeaza vectorii proprii pentru consistenta
        """
        self.X = X  # Matricea initiala de date.
        self.X_std = self.standardizareDate()  # Standardizarea datelor.
        self.cov_matrix = self.calculCovarianta()  # Matricea de covarianta.
        self.eigenvalues, self.eigenvectors = self.descompunereValoriVectoriProprii()  # Valori si vectori proprii.

        # Regularizare pentru consistenta
        if regularizare:
            self.regularizareVectoriProprii()

        # Calculul scorurilor PCA
        self.scoruri = self.calculScoruri()

        # Calculul incarcarilor factorilor
        self.factor_loadings = self.calculFactoriLoadings()

    def standardizareDate(self):
        """
        Standardizeaza datele astfel incat fiecare variabila sa aiba media 0 si deviatia standard 1
        """
        return (self.X - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)

    def calculCovarianta(self):
        """
        Calculeaza matricea de covarianta pentru datele standardizate
        """
        return np.cov(self.X_std, rowvar=False)

    def descompunereValoriVectoriProprii(self):
        """
        Calculeaza valorile si vectorii proprii ai matricei de covarianta
        - eigenvalues: valorile proprii sortate descrescator
        - eigenvectors: vectorii proprii corespunzatori.
        """
        values, vectors = np.linalg.eigh(self.cov_matrix)
        sorted_indices = np.argsort(values)[::-1]  # Sorteaza valorile proprii descrescator
        return values[sorted_indices], vectors[:, sorted_indices]

    def regularizareVectoriProprii(self):
        """
        Regularizeaza vectorii proprii pentru consistenta semnului
        """
        for j in range(self.eigenvectors.shape[1]):
            if abs(np.min(self.eigenvectors[:, j])) > abs(np.max(self.eigenvectors[:, j])):
                self.eigenvectors[:, j] *= -1

    def calculScoruri(self):
        """
        Calculeaza scorurile PCA (proiectia observatiilor pe componentele principale)
        """
        return self.X_std @ self.eigenvectors

    def calculFactoriLoadings(self):
        """
        Calculeaza incarcarile factorilor (relatia dintre variabilele originale si componentele principale)
        """
        return self.eigenvectors * np.sqrt(self.eigenvalues)

    def getModelStdandardizate(self):
        """
        Returneaza datele standardizate
        """
        return self.X_std

    def getValoriProprii(self):
        """
        Returneaza valorile proprii
        """
        return self.eigenvalues

    def getVectoriProprii(self):
        """
        Returneaza vectorii proprii
        """
        return self.eigenvectors

    def getScoruriPCA(self):
        """
        Returneaza scorurile PCA
        """
        return self.scoruri

    def getFactorLoadings(self):
        """
        Returneaza incarcarile factorilor
        """
        return self.factor_loadings

    def getComunalitati(self):
        """
        Calculeaza comunalitatile (proportia variabilitatii explicate de componentele principale pentru fiecare variabila)
        """
        factor_loadings_squared = self.factor_loadings ** 2
        return np.cumsum(factor_loadings_squared, axis=1)

    def getCalitateaObservatiilor(self):
        """
        Calculeaza calitatea reprezentarii fiecarei observatii in spatiul componentelor principale
        """
        total_variance = np.sum(self.scoruri ** 2, axis=1)
        return np.transpose(self.scoruri.T ** 2 / total_variance)

    def getContributiaObservatiilor(self):
        """
        Calculeaza contributia fiecarei observatii la variabilitatea fiecarei componente principale
        """
        return (self.scoruri ** 2) / (self.eigenvalues * self.X.shape[0])
