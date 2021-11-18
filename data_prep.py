import pandas as pd
from sklearn.preprocessing import LabelEncoder

def data_prep(df: pd.DataFrame):
    #ignorar atributo classe
    df_prep = df[df.columns[:-1]].copy()
    le = LabelEncoder()
    for col in df_prep.columns:
        #extrai a moda da coluna
        moda = df_prep[col].mode().iloc[0]
        
        #substitui valores nulos com o valor encontrado para moda
        df_prep[col].replace("?", moda, inplace=True)
        
        #tenta convertar tipo da coluna para numerico
        #pois o valor '?' faz com que colunas numericas fossem interpretadas
        #como colunas do tipo string
        try:
            df_prep[col] = pd.to_numeric(df_prep[col])
        
        #caso coluna nao seja numerica mesmo
        #codificar suas categorias em valores numericos
        except:
            df_prep[col] = le.fit_transform(df_prep[col])
            df_prep[col] = pd.to_numeric(df_prep[col])
    return df_prep