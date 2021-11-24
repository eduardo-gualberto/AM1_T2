from model import kmeans_model
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# sum squared error
# usado para fazer a plotagem dos dados
# SSEs = {}
# s_scores = []
# for k in range(2, 8):
#     model, df = kmeans_model(k)
#     model = model.fit(df)
#     preds = model.predict(df)
#     SSEs[k] = model.inertia_
#     s_scores.append(silhouette_score(df, preds))
# print(s_scores)
# plt.figure()
# plt.plot(list(SSEs.keys()), list(SSEs.values()))
# plt.xlabel("Number of cluster")
# plt.ylabel("Silhouette Score")
# for i, label in enumerate(SSEs):
#     plt.annotate(format(SSEs[label], ".2E"), (label, SSEs[label]))
# plt.show()
SSEs = {}
for k in range(2, 8):
    model, df = kmeans_model(k)
    model = model.fit(df)
    preds = model.predict(df)
    SSEs[k] = silhouette_score(df, preds)
plt.figure()
plt.plot(list(SSEs.keys()), list(SSEs.values()))
plt.xlabel("Number of cluster")
plt.ylabel("Silhouette Score")
for i, label in enumerate(SSEs):
    plt.annotate(format(SSEs[label], ".4f"), (label, SSEs[label]))
plt.show()
