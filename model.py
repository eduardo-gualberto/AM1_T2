from sklearn.cluster import KMeans
import pandas as pd
from data_prep import data_prep

#abrir arquivo de dados
raw_df = raw_df = pd.read_csv('dataset_194_eucalyptus.csv')

#etapa de preparação dos dados
df = data_prep(raw_df)

kmeans = KMeans(n_clusters=5).fit(df)