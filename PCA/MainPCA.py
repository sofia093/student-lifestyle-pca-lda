import os
import pandas as pd
import numpy as np
from PCA import PCA
import graficePCA as gr
import matplotlib.pyplot as plt

# Folderul pentru rezultate
results_folder = "../dataOUT/PCA_results/"

# Pregatirea si standardizarea datelor
# Citirea fisierului Excel
file_path = "../../../../../PycharmProjects/student-lifestyle-pca-lda/dataIN/lifestyle_studenti_data.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# Selectarea variabilelor numerice pentru analiza PCA
variabile_numerice = ['Ore_Studiu', 'Ore_Activitati_Extracuriculare', 'Ore_Somn',
                      'Ore_Social', 'Ore_Activ', 'GPA']
X = df[variabile_numerice].values

# Aplicarea PCA folosind clasa definita
pca_model = PCA(X)

# Matricea de corelatii
# Calculul si salvarea matricei de corelatii
matrice_corelatii = df[variabile_numerice].corr()
matrice_corelatii.to_excel(f"{results_folder}/matrice_corelatii.xlsx", float_format="%.2f", index_label="Variabile")
gr.corelograma(matrix=matrice_corelatii, title="Corelograma Variabilelor")
gr.salveazaGrafic(f"{results_folder}/corelograma_variabile.png")

# Valorile proprii si selectarea componentelor principale
# Calculul valorilor proprii
valorile_proprii = pca_model.getValoriProprii()
print("Valorile proprii:", valorile_proprii)

# Criteriul Kaiser
criteriul_kaiser = np.sum(valorile_proprii >= 1)
print(f"Numarul de componente principale conform criteriului Kaiser: {criteriul_kaiser}")

# Procentul de acoperire
var_explicata_cumulativ = np.cumsum(valorile_proprii) / np.sum(valorile_proprii)
procent_acoperire = 0.8
criteriul_acoperire = np.sum(var_explicata_cumulativ <= procent_acoperire)
print(f"Numarul de componente principale semnificative conform procentului de acoperire (80%): {criteriul_acoperire}")

# Scree Plot (Criteriul Cattel)
gr.componentePrincipale(eigenvalues=valorile_proprii, title="Componentele Principale (Scree Plot)")
gr.salveazaGrafic(f"{results_folder}/componente_principale_scree_plot.png")

# Salvam valorile proprii si varianta explicata
valorile_proprii_df = pd.DataFrame({
    "Valori proprii": valorile_proprii,
    "Varianta explicata (%)": (valorile_proprii / np.sum(valorile_proprii)) * 100,
    "Varianta explicata cumulativ (%)": var_explicata_cumulativ * 100
})
valorile_proprii_df.to_excel(f"{results_folder}/valorile_proprii.xlsx", index_label="Componenta")

# Comunalitatile
# Calculul comunalitatilor
comunalitati = pca_model.getComunalitati()
comunalitati_df = pd.DataFrame(data=comunalitati,
                               columns=[f"C{i+1}" for i in range(comunalitati.shape[1])],
                               index=variabile_numerice)
# Salvam comunalitatile in fisier Excel
comunalitati_df.to_excel(f"{results_folder}/comunalitati.xlsx", index_label="Variabile")
gr.corelograma(matrix=comunalitati_df, title="Corelograma Comunalitatilor")
gr.salveazaGrafic(f"{results_folder}/corelograma_comunalitati.png")

# Incarcarile factorilor (Factor Loadings)
# Generarea incarcarilor factorilor
factor_loadings = pca_model.getFactorLoadings()
factor_loadings_df = pd.DataFrame(data=factor_loadings,
                                  columns=[f"C{i+1}" for i in range(factor_loadings.shape[1])],
                                  index=variabile_numerice)
factor_loadings_df.to_excel(f"{results_folder}/factor_loadings.xlsx", index_label="Variabile")

# Cercul corelatiilor
# Generarea cercului corelatiilor
gr.cercCorelatie(matrix=factor_loadings_df, V1=0, V2=1, title="Cercul Corelatiilor")
gr.salveazaGrafic(f"{results_folder}/cerc_corelatii.png")

# Calitatea Reprezentarii Observatiilor
# Calculul calitatii reprezentarii observatiilor
qual_obs = pca_model.getCalitateaObservatiilor()
qual_obs_df = pd.DataFrame(data=qual_obs,
                           columns=[f"C{i+1}" for i in range(qual_obs.shape[1])],
                           index=df.index)
# Salvam calitatea reprezentarii in Excel
qual_obs_stats = qual_obs_df.describe()
qual_obs_stats.to_excel(f"{results_folder}/calitate_reprezentare_stats.xlsx")

# Grafic complet pentru calitatea reprezentarii observatiilor
plt.figure(figsize=(25, 12))
gr.linkIntesity(matrix=qual_obs_df, title='Calitatea Reprezentarii Observatiilor (Toate)')
gr.salveazaGrafic(f"{results_folder}/calitate_reprezentare_toate.png")

# Grafic pentru subset de observatii
qual_obs_sorted = qual_obs_df.sort_values(by="C1", ascending=False)
qual_obs_subset = qual_obs_sorted.head(50)
plt.figure(figsize=(20, 10))
gr.linkIntesity(matrix=qual_obs_subset, title='Calitatea Reprezentarii Observatiilor (Subset)')
gr.salveazaGrafic(f"{results_folder}/calitate_reprezentare_subset.png")

# 4.8. Contributia observatiilor
# Calculul contributiei observatiilor la variatia axelor
obs_contrib = pca_model.getContributiaObservatiilor()
obs_contrib_df = pd.DataFrame(data=obs_contrib,
                              columns=[f"C{i+1}" for i in range(obs_contrib.shape[1])],
                              index=df.index)
obs_contrib_df.to_excel(f"{results_folder}/contributii_observatii.xlsx", index_label="Observatii")

# Grafic complet pentru contributia observatiilor
plt.figure(figsize=(25, 12))
gr.linkIntesity(matrix=obs_contrib_df, title='Contributia Observatiilor la Variatia Axelor (Toate)')
gr.salveazaGrafic(f"{results_folder}/contributii_observatii_toate.png")

# Grafic pentru contributia observatiilor (subset)
obs_contrib_subset = obs_contrib_df.head(50)
plt.figure(figsize=(20, 10))
gr.linkIntesity(matrix=obs_contrib_subset, title='Contributia Observatiilor la Variatia Axelor (Subset)')
gr.salveazaGrafic(f"{results_folder}/contributii_observatii_subset.png")

print(f"Analiza PCA este completa. Rezultatele au fost salvate in folderul {results_folder}")
