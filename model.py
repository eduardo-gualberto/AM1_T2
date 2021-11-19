from sklearn.cluster import KMeans
import pandas as pd
from data_prep import data_prep

def kmeans_model(k: int):
    #abrir arquivo de dados
    raw_df = pd.read_csv('dataset_194_eucalyptus.csv')

    #etapa de preparação dos dados
    df = data_prep(raw_df)

    kmeans = KMeans(n_clusters=k).fit(df)
    return kmeans