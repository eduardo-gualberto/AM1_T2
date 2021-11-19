from model import kmeans_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# sum squared error
# usado para fazer a plotagem dos dados
SSEs = {}
for k in range(1, 11):
    model = kmeans_model(k)
    SSEs[k] = model.inertia_

plt.figure()
plt.plot(list(SSEs.keys()), list(SSEs.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
for i, label in enumerate(SSEs):
    plt.annotate(format(SSEs[label], ".2E"), (label, SSEs[label]))
plt.show()