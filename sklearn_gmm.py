# filename: gmm_sklearn.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
from pathlib import Path
from sklearn.mixture import GaussianMixture

class SklearnGMMAnalyzer:
    def __init__(self, csv_path, results_root='results', n_components=3, random_state=42):
        self.csv_path = csv_path
        self.n_components = n_components
        self.random_state = random_state
        self.data = None
        self.numeric_data = None
        self.model = None

        base = os.path.basename(csv_path)
        name, _ = os.path.splitext(base)
        timestamp = time.strftime("%m-%d_%H-%M-%S")
        self.output_dir = os.path.join(results_root, f"{timestamp}_{name}")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.log_file = os.path.join(self.output_dir, "log.txt")

    def log(self, message, start_time=None):
        timestamp = time.strftime("[%H:%M:%S]")
        duration = f" (+{time.time() - start_time:.2f}s)" if start_time else ""
        full_message = f"{timestamp} {message}{duration}"
        print(full_message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(full_message + "\n")

    def run(self):
        try:
            print("sklearn_gmm.py")
            self.log("🔄 Запуск GMM-аналізу...")
            
            t0 = time.time()
            self.load_data()
            self.log("📥 Завантаження даних завершено.", t0)

            t1 = time.time()
            self.fit_model()
            self.log("🔍 Навчання моделі GMM завершено.", t1)

            t2 = time.time()
            self.save_model_params()
            self.log("🧠 Збереження параметрів завершено.", t2)

            t3 = time.time()
            self.save_results()
            self.log("💾 Збереження результатів завершено.", t3)

            t4 = time.time()
            self.generate_plots()
            self.log("📊 Генерація графіків завершена.", t4)

            self.log("✅ Аналіз завершено успішно.", t0)
        except Exception as e:
            self.log(f"❌ ПОМИЛКА: {e}")
            raise

    def load_data(self):
        self.log(f"Читання CSV: {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)
        self.numeric_data = self.data[["X", "Y", "Z"]].dropna()
        self.log(f"Отримано {len(self.numeric_data)} валідних зразків.")

    def fit_model(self):
        self.log(f"Початок навчання GMM з n_components={self.n_components}, random_state={self.random_state}")
        self.model = GaussianMixture(n_components=self.n_components, random_state=self.random_state)
        self.model.fit(self.numeric_data)
        self.log("✅ GMM навчено.")
        
        self.log(f"Середні значення кластерів (means):\n{self.model.means_}")
        self.log(f"Коваріації кластерів (covariances):\n{self.model.covariances_}")
        self.log(f"Ваги кластерів (weights): {self.model.weights_}")

        clusters = self.model.predict(self.numeric_data)
        self.data["Cluster"] = clusters

        log_probs = self.model.score_samples(self.numeric_data)
        self.data["Log_Probability"] = log_probs

        self.log("🔢 Кластери та log-імовірності додано до даних.")

        new_samples = self.model.sample(10)[0]
        self.new_samples_df = pd.DataFrame(new_samples, columns=["X", "Y", "Z"])
        self.log("🆕 Згенеровано 10 нових зразків з GMM.")

    def save_model_params(self):
        params = {
            "n_components": self.n_components,
            "random_state": self.random_state,
            "means": self.model.means_.tolist(),
            "covariances": self.model.covariances_.tolist(),
            "weights": self.model.weights_.tolist(),
        }
        path = os.path.join(self.output_dir, "gmm_params.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=4)
        self.log(f"🧠 Параметри моделі збережені в {path}")

    def save_results(self):
        base = os.path.basename(self.csv_path)
        name, _ = os.path.splitext(base)
        clusters_csv = os.path.join(self.output_dir, "processed_data.csv") #f"{name}_with_clusters.csv"
        generated_csv = os.path.join(self.output_dir, f"{name}_generated_samples.csv")
        self.data.to_csv(clusters_csv, index=False)
        self.new_samples_df.to_csv(generated_csv, index=False)
        self.log(f"📄 CSV з кластерами збережено як {clusters_csv}")
        self.log(f"📄 CSV з новими зразками збережено як {generated_csv}")

    def generate_plots(self):
        cluster_plot_path = os.path.join(self.output_dir, "plot_clusters.png")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x="X", y="Y", hue=self.data["Cluster"].map(lambda x: f"Кластер_{x+1}"),
                        data=self.data, palette='viridis')
        plt.title("📌 Візуалізація кластерів")
        plt.legend(title="Кластери")
        plt.savefig(cluster_plot_path)
        plt.close()
        self.log("📍 Графік кластерів збережено.")

        ll_plot_path = os.path.join(self.output_dir, "plot_log_likelihood.png")
        plt.figure(figsize=(8, 4))
        plt.plot(self.data["Log_Probability"], color='blue')
        plt.title("📉 Log-Likelihood по зразках")
        plt.xlabel("Індекс зразка")
        plt.ylabel("Log Probability")
        plt.savefig(ll_plot_path)
        plt.close()
        self.log("📈 Графік log-likelihood збережено.")

        hist_plot_path = os.path.join(self.output_dir, "plot_histogram.png")
        plt.figure(figsize=(6, 4))
        self.data["Cluster"].value_counts().sort_index().plot(kind="bar", color='green')
        plt.title("📊 Розподіл зразків по кластерам")
        plt.xlabel("Кластер")
        plt.ylabel("Кількість зразків")
        plt.savefig(hist_plot_path)
        plt.close()
        self.log("📊 Гістограма кластерів збережена.")

        heatmap_path = os.path.join(self.output_dir, "plot_heatmap.png")
        plt.figure(figsize=(6, 5))
        sns.heatmap(self.numeric_data.corr(), annot=True, cmap='coolwarm')
        plt.title("🔥 Теплова карта (кореляція між сенсорами)")
        plt.savefig(heatmap_path)
        plt.close()
        self.log("🔥 Теплова карта збережена.")
