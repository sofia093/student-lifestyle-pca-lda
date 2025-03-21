import os
import pandas as pd
import numpy as np
import utilsDA as utils
import graficeDA as graphics
import sklearn.metrics as metrics

try:
    # Pregatirea Datelor
    # Se incarca fisierul cu date, se verifica si inlocuiesc valorile lipsa, iar variabilele categorice sunt codificate.
    file_1 = '../dataIN/lifestyle_studenti_data.xlsx'
    da_results_dir = '../dataOUT/DA_results/'
    os.makedirs(da_results_dir, exist_ok=True)

    table_1 = pd.read_excel(file_1, index_col=0)
    utils.replace_na_df(table_1)

    vars = np.array(table_1.columns)
    var_p, var_c = vars[:-1], vars[-1]
    utils.codify(table_1, vars[:0])

    x, y = table_1[var_p].values, table_1[var_c].values
    n, m = x.shape

    # Matricele SSW, SSB si SST
    # Se calculeaza cele trei matrici care definesc dispersia in analiza discriminanta.
    g, ng, xg, sst, ssb, ssw = utils.dispersion(x, y)
    q = len(g)

    # Salvare matrice de dispersie pentru analiza ulterioara
    pd.DataFrame(ssw).to_csv(f'{da_results_dir}ssw_matrix.csv')
    pd.DataFrame(ssb).to_csv(f'{da_results_dir}ssb_matrix.csv')
    pd.DataFrame(sst).to_csv(f'{da_results_dir}sst_matrix.csv')

    # Puterea de Discriminare a Variabilelor
    # Se calculeaza cat de bine fiecare variabila contribuie la separarea grupurilor.
    l_x, p_values = utils.discrim_power(ssb, ssw, n, q)

    discrimination_power_df = pd.DataFrame({
        'Discrimination_power': l_x,
        'p_value': np.round(p_values, 2)
    }, index=var_p)

    discrimination_power_df.to_csv(f'{da_results_dir}Discrimination_power.csv')

    # Matricea de Confuzie si Performanta Modelului
    # Se aplica LDA, se realizeaza clasificarea si se evalueaza performanta modelului.
    alpha, l, u = utils.lda(sst, ssb, n, q)
    z, zg = x @ u, xg @ u

    f, f0, f0_b = utils.classification_functions(z, zg, ng)
    classification_LDA = utils.predict(z, f, f0, g)
    classification_Bayes = utils.predict(z, f, f0_b, g)

    # Evaluarea performantelor modelului prin acuratete si scor Cohen Kappa
    accuracy_global = pd.DataFrame({
        'Accuracy': [100 - sum(classification_LDA != y) * 100 / n,
                     100 - sum(classification_Bayes != y) * 100 / n],
        'Cohen_Kappa': [metrics.cohen_kappa_score(y, classification_LDA),
                        metrics.cohen_kappa_score(y, classification_Bayes)]
    }, index=['LDA', 'Bayes'])

    accuracy_global.to_csv(f'{da_results_dir}Accuracy.csv')

    # Salvare matrice de confuzie pentru a evalua greselile de clasificare
    utils.discrim_accuracy(y, classification_LDA, g).to_csv(f'{da_results_dir}Mat_conf_LDA.csv')
    utils.discrim_accuracy(y, classification_Bayes, g).to_csv(f'{da_results_dir}Mat_conf_Bayes.csv')

    # Separarea Grupurilor si Centrii Lor
    # Se genereaza un scatter plot pentru a vizualiza separarea grupurilor in spatiul discriminant.
    if len(l) > 1:
        graphics.plot_scatter_groups(
            x=z[:, 0], y=z[:, 1], g=y,
            x1=zg[:, 0], y1=zg[:, 1], g1=g,
            title="Scatter Plot of Discriminant Scores",
            lx=f'z1 ({round(l[0] * 100 / sum(l), 2)}%)',
            ly=f'z2 ({round(l[1] * 100 / sum(l), 2)}%)',
            save_path=f'{da_results_dir}scatter_groups.png'
        )

    # Salvarea coordonatelor centrilor grupurilor
    pd.DataFrame(zg, index=g).to_csv(f'{da_results_dir}group_centers.csv')

    # Interpretarea Rezultatelor
    # Se analizeaza distributia scorurilor discriminante pe fiecare axa.
    for i in range(len(l)):
        graphics.plot_distribution(
            z[:, i], y, g,
            title=f'Distribution Axis {i + 1}',
            save_path=f'{da_results_dir}distribution_z{i + 1}.png'
        )

    print("Analiza finalizata cu succes!")

except Exception as ex:
    print(f"Eroare: {ex}")
