# filename: my_gmm.py

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# новий файл = n  в сек 
# сохраняю модель
# роблю предікт

#екран з моделями (скачування моделей)
#запуск моделей з різними моделями(готовими і ні)
#beta-flyer
#rocket launh data imu

class GMM:
    # Ініціалізація параметрів моделі
    def __init__(self, n_components, max_iters=100, tol=1e-6, random_state=42, logger=None, reg_cov=1e-6):
        self.n_components = n_components  # Кількість компонентів (кластерів) в моделі
        self.max_iters = max_iters  # Максимальна кількість ітерацій для алгоритму
        self.tol = tol  # Допустиме значення відхилення для припинення ітерацій
        self.random_state = random_state  # Встановлення випадкового стану для відтворюваності результатів
        self.logger = logger  # Логер для запису процесу
        self.reg_cov = reg_cov  # Регуляризація для ковариацій, щоб уникнути їх виродження
        self.trained = False  # Прапорець, який вказує на те, що модель навчена

    # Ініціалізація параметрів (середніх значень, ковариацій та ваг)
    def initialize_params(self, X):
        n_samples, n_features = X.shape  # Отримуємо розміри вибірки
        np.random.seed(self.random_state)  # Фіксування випадкового стану для відтворюваності результатів
        self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]  # Вибір початкових середніх значень
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])  # Ініціалізація ковариацій (ідентичні матриці)
        self.weights = np.ones(self.n_components) / self.n_components  # Рівні ваги для кожного компонента
        self.resp = np.zeros((n_samples, self.n_components))  # Матриця відповідальностей (яка компонента є більш ймовірною)

    # Функція для обчислення ймовірності за допомогою гауссівського розподілу
    def gaussian_pdf(self, X, mean, cov):
        n = X.shape[1]  # Кількість ознак
        diff = X - mean  # Різниця між даними та середнім
        try:
            inv_cov = np.linalg.inv(cov)  # Обчислюємо зворотну матрицю ковариацій
        except np.linalg.LinAlgError:
            return np.zeros(X.shape[0])  # Якщо матриця ковариацій вироджена (необратна), повертаємо нулі
        exp_term = np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=1))  # Експоненціальний компонент
        denom = np.sqrt((2 * np.pi) ** n * np.linalg.det(cov))  # Нормалізаційний множник
        return exp_term / denom  # Повертаємо ймовірність

    # E-етап (обчислення відповідальностей для кожного компонента)
    def e_step(self, X):
        for i in range(self.n_components):  # Для кожного компонента
            self.resp[:, i] = self.weights[i] * self.gaussian_pdf(X, self.means[i], self.covariances[i])  # Обчислюємо відповідальність
        self.resp /= self.resp.sum(axis=1, keepdims=True)  # Нормалізація відповідальностей (сума має бути 1)
        if np.any(np.isnan(self.resp)):  # Перевірка на NaN в матриці відповідальностей
            raise ValueError("NaN values found in responsibility matrix.")  # Якщо є NaN, викидаємо помилку

    # M-етап (оновлення параметрів моделі)
    def m_step(self, X):
        Nk = self.resp.sum(axis=0)  # Сума відповідальностей по кожному компоненту
        self.weights = Nk / X.shape[0]  # Оновлюємо ваги (пропорція відповідальностей для кожного компонента)
        self.means = (self.resp.T @ X) / Nk[:, np.newaxis]  # Оновлюємо середні значення
        for i in range(self.n_components):
            diff = X - self.means[i]  # Різниця між даними та новим середнім
            self.covariances[i] = (self.resp[:, i][:, np.newaxis] * diff).T @ diff / Nk[i]  # Оновлюємо ковариації
            self.covariances[i] += np.eye(X.shape[1]) * self.reg_cov  # Додаємо регуляризацію до ковариацій
            if np.any(np.isnan(self.covariances[i])):  # Перевірка на NaN в ковариаційній матриці
                raise ValueError(f"NaN values found in covariance matrix for component {i}.")  # Якщо є NaN, викидаємо помилку

    # Основна функція для навчання моделі
    def fit(self, X):
        if self.logger:
            self.logger(f"📌 Початок навчання GMM з n_components={self.n_components}, random_state={self.random_state}")

        self.initialize_params(X)  # Ініціалізуємо параметри моделі

        # Основний цикл ітерацій
        for iteration in range(self.max_iters):
            prev_means = self.means.copy()  # Зберігаємо попередні значення середніх
            try:
                self.e_step(X)  # E-етап
                self.m_step(X)  # M-етап
            except ValueError as e:
                if self.logger:
                    self.logger(f"❌ {e}")  # Логування помилки, якщо виникає NaN
                break

            if self.logger:
                self.logger(f"🔁 Ітерація {iteration+1}")
                self.logger(f"Середні значення кластерів (means): {self.means.tolist()}")
                self.logger(f"Ваги кластерів (weights): {self.weights.tolist()}")

            # Перевірка на конвергенцію
            if np.linalg.norm(self.means - prev_means) < self.tol:
                if self.logger:
                    self.logger(f"✅ Конвергенція досягнута на ітерації {iteration+1}")
                break

        self.trained = True  # Після навчання встановлюємо прапорець

    # Прогнозування для нових даних
    def predict(self, X):
        if not self.trained:
            raise Exception("Model is not trained yet.")  # Якщо модель не навчена, викидаємо помилку
        likelihoods = np.array([  # Обчислюємо ймовірності для кожного компонента
            self.weights[i] * self.gaussian_pdf(X, self.means[i], self.covariances[i])
            for i in range(self.n_components)
        ]).T
        return np.argmax(likelihoods, axis=1)  # Повертаємо індекс компонента з найбільшою ймовірністю

    # Обчислення логарифму ймовірності для кожного зразка
    def score_samples(self, X):
        likelihoods = np.zeros(X.shape[0])  # Масив для ймовірностей
        for i in range(self.n_components):
            likelihoods += self.weights[i] * self.gaussian_pdf(X, self.means[i], self.covariances[i])  # Сума ймовірностей для кожного компонента
        return np.log(np.clip(likelihoods, a_min=1e-300, a_max=None))  # Логарифм ймовірностей, з обмеженням на мінімум

    # Отримання параметрів моделі
    def get_params(self):
        return {
            "means": self.means.tolist(),  # Середні значення компонентів
            "covariances": [c.tolist() for c in self.covariances],  # Ковариаційні матриці компонентів
            "weights": self.weights.tolist()  # Ваги компонентів
        }




class MyGMMAnalyzer:
    def __init__(self, csv_path, results_root='results', n_components=3):
        self.csv_path = csv_path
        self.n_components = n_components
        self.data = None
        self.numeric_data = None
        self.model = None

        base = os.path.basename(csv_path)
        name, _ = os.path.splitext(base)
        timestamp = time.strftime("%m-%d_%H-%M-%S")
        self.output_dir = os.path.join(results_root, f"{timestamp}_{name}_mygmm")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = os.path.join(self.output_dir, "log.txt")

    def log(self, message):
        timestamp = time.strftime("[%H:%M:%S]")
        full_message = f"{timestamp} {message}"
        print(full_message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(full_message + "\n")

    def run(self):
        try:
            print(" my_gmm.py")
            print(self.csv_path)
            self.log("🔄 Запуск MyGMM-аналізу...")
            self.load_data()
            self.log("📥 Завантаження даних завершено.")
            self.fit_model()
            self.log("🔍 Навчання моделі завершено.")
            self.save_results()
            self.log("💾 Результати збережено.")
            self.generate_plots()
            self.log("📊 Візуалізація завершена.")
        except Exception as e:
            self.log(f"❌ ПОМИЛКА: {e}")
            raise

    def load_data(self):
        self.log(f"Читання CSV: {self.csv_path}")
        self.data = pd.read_csv(self.csv_path)
        self.numeric_data = self.data[["X", "Y", "Z"]].dropna().values
        self.log(f"Отримано {len(self.numeric_data)} валідних зразків.")

    def fit_model(self):
        self.model = GMM(n_components=self.n_components, random_state=42, logger=self.log)
        self.model.fit(self.numeric_data)
        self.data["Cluster"] = self.model.predict(self.numeric_data)
        self.data["Log_Probability"] = self.model.score_samples(self.numeric_data)

    def save_results(self):
        base = os.path.basename(self.csv_path)
        name, _ = os.path.splitext(base) 
        clusters_csv = os.path.join(self.output_dir, "processed_data.csv") #f"{name}_with_clusters.csv"
        self.data.to_csv(clusters_csv, index=False)
        self.log(f"📄 CSV з кластерами збережено як {clusters_csv}")

        # Збереження параметрів GMM у JSON
        gmm_params_path = os.path.join(self.output_dir, "gmm_params.json")
        with open(gmm_params_path, "w", encoding="utf-8") as f:
            json.dump(self.model.get_params(), f, indent=2)
        self.log(f"📁 Параметри моделі збережені в {gmm_params_path}")

    def generate_plots(self):
        cluster_plot_path = os.path.join(self.output_dir, "plot_clusters.png")
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x="X", y="Y", hue=self.data["Cluster"].map(lambda x: f"Кластер_{x+1}"),
                        data=self.data, palette='viridis')
        plt.title("📌 Візуалізація кластерів")
        plt.legend(title="Кластери")
        plt.savefig(cluster_plot_path)
        plt.close()

        ll_plot_path = os.path.join(self.output_dir, "plot_log_likelihood.png")
        plt.figure(figsize=(8, 4))
        plt.plot(self.data["Log_Probability"], color='blue')
        plt.title("📉 Log-Likelihood по зразках")
        plt.xlabel("Індекс зразка")
        plt.ylabel("Log Probability")
        plt.savefig(ll_plot_path)
        plt.close()

        hist_plot_path = os.path.join(self.output_dir, "plot_histogram.png")
        plt.figure(figsize=(6, 4))
        self.data["Cluster"].value_counts().sort_index().plot(kind="bar", color='green')
        plt.title("📊 Розподіл зразків по кластерам")
        plt.xlabel("Кластер")
        plt.ylabel("Кількість зразків")
        plt.savefig(hist_plot_path)
        plt.close()

        heatmap_path = os.path.join(self.output_dir, "plot_heatmap.png")
        plt.figure(figsize=(6, 5))
        sns.heatmap(pd.DataFrame(self.numeric_data, columns=["X", "Y", "Z"]).corr(), annot=True, cmap='coolwarm')
        plt.title("🔥 Теплова карта (кореляція між сенсорами)")
        plt.savefig(heatmap_path)
        plt.close()
