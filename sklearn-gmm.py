import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Завантаження даних
data = pd.read_csv("sensor_data_phone1.csv")

# Фільтруємо тільки числові колонки (X, Y, Z)
numeric_data = data[["X", "Y", "Z"]].dropna()

# Навчання GMM з 3 компонентами
n_components = 3
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(numeric_data)

# 1. Кластеризація: передбачення кластерів
clusters = gmm.predict(numeric_data)
data["Cluster"] = clusters

# 2. Оцінка щільності ймовірності
log_probs = gmm.score_samples(numeric_data)
data["Log_Probability"] = log_probs

# 3. Генерація нових точок
new_samples = gmm.sample(10)[0]
new_samples_df = pd.DataFrame(new_samples, columns=["X", "Y", "Z"])

# Візуалізація кластерів
plt.scatter(numeric_data["X"], numeric_data["Y"], c=clusters, cmap='viridis', alpha=0.5)
plt.scatter(new_samples[:, 0], new_samples[:, 1], color='red', marker='x', label='Generated Points')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# Збереження результатів
data.to_csv("sensor_data_with_clusters.csv", index=False)
new_samples_df.to_csv("generated_samples.csv", index=False)
