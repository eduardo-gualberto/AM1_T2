import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from data_prep import data_prep
from model import kmeans_model

raw_df = pd.read_csv('dataset_194_eucalyptus.csv')
df = data_prep(raw_df)
df['class'] = raw_df['Utility']
df['pred'] = pd.Series(kmeans_model(5).labels_)

# classes = list(set(df["Utility"].to_list()))
# colors = ['red', 'yellow', 'green', 'blue', 'orange']
# mapping = {}
# for i in range(len(classes)):
#     mapping[classes[i]] = colors[i]

# sb.pairplot(df, hue="pred", vars=df.columns[-7:-2], palette='tab10')
# plt.show()