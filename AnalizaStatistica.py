import pandas as pd

file_path = "../../../../PycharmProjects/student-lifestyle-pca-lda/dataIN/lifestyle_studenti_data.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

statistici_descriptive = df.describe()
statistici_descriptive.to_excel("./dataOUT/statistici_descriptive.xlsx")
print("Statisticile descriptive au fost salvate in './dataOUT/statistici_descriptive.xlsx'.")
