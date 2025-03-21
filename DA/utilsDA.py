import numpy as np
import pandas as pd
import collections as co
import scipy.stats as sts

# Inlocuire valori lipsa in DataFrame
def replace_na_df(df):
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# Codificare variabile categorice
def codify(df, vars):
    for var in vars:
        df[var] = pd.Categorical(df[var]).codes
    return df

# Calcul dispersie totala, intre si in interiorul grupurilor
def dispersion(x, y):
    n, m = np.shape(x)
    means = np.mean(x, axis=0)
    counter = co.Counter(y)
    g = np.array(list(counter.keys()))
    ng = np.array(list(counter.values()))
    xg = np.array([np.mean(x[y == group, :], axis=0) for group in g])
    xg_med = xg - means
    sst = n * np.cov(x, rowvar=False, bias=True)
    ssb = np.transpose(xg_med) @ np.diag(ng) @ xg_med
    ssw = sst - ssb
    return g, ng, xg, sst, ssb, ssw

# Puterea de discriminare
def discrim_power(ssb, ssw, n, q):
    r = (n - q) / (q - 1)
    f = r * np.diag(ssb) / np.diag(ssw)
    p_values = 1 - sts.f.cdf(f, q - 1, n - q)
    return f, p_values

# Analiza Discriminanta Liniara (LDA)
def lda(sst, ssb, n, q):
    cov_inv = np.linalg.inv(sst)
    h = cov_inv @ ssb
    eigenvalues, eigenvectors = np.linalg.eig(h)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues[:q - 1], eigenvalues[:q - 1], eigenvectors[:, :q - 1]

# Functii de clasificare LDA si Bayes
def classification_functions(z, zg, ng):
    f = zg @ np.diag(1.0 / np.var(z, axis=0))
    f0 = -0.5 * np.sum(f * zg, axis=1)
    f0_b = f0 + np.log(ng / len(z))
    return f, f0, f0_b

# Prezicere clasa pe baza functiilor de clasificare
def predict(z, f, f0, g):
    scores = z @ f.T + f0
    return g[np.argmax(scores, axis=1)]

# Matrice de confuzie
def discrim_accuracy(y, pred, g):
    conf_matrix = pd.DataFrame(np.zeros((len(g), len(g))), index=g, columns=g)
    for true, pred_class in zip(y, pred):
        conf_matrix.loc[true, pred_class] += 1
    return conf_matrix
