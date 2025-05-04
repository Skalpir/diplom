# filename: my_gmm.py

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class GMM:
    # Ініціалізація параметрів моделі
    def __init__(
            self, 
            n_components, 
            max_iters=100, 
            tol=1e-6, 
            random_state=42, 
            logger=None, 
            reg_cov=1e-6,
            covariance_type='full',
            init_params='kmeans', #'kmeans' або 'random' abo 'k-means++'
            means_init=None,
            precisions_init=None,
            weights_init=None,
            #n_init=1,  # Кількість запусків з різними ініціалізаціями, модель обирає кращий #TODO заготовочка для авто-визначення кількості кластерів

            ):
        self.n_components = n_components  # Кількість компонентів (кластерів) в моделі
        self.max_iters = max_iters  # Максимальна кількість ітерацій для алгоритму
        self.tol = tol  # Допустиме значення відхилення для припинення ітерацій
        self.random_state = random_state  # Встановлення випадкового стану для відтворюваності результатів
        self.logger = logger  # Логер для запису процесу
        self.reg_cov = reg_cov  # Регуляризація для ковариацій, щоб уникнути їх виродження
        self.trained = False  # Прапорець, який вказує на те, що модель навчена
        self.covariance_type = covariance_type  # Тип ковариацій (повна, діагональна тощо)
        self.init_params = init_params  # Метод ініціалізації параметрів (kmeans, random)
        #параметри для до-навчання моделі, по суті для IMU , тому що ІМУ працює через збереження стану моделі і завантаженні її знову для нових точок
        self.means_init = means_init  # Початкові середні значення (якщо є)
        self.precisions_init = precisions_init # Ковариаційні матриці (якщо є)
        self.weights_init = weights_init # Початкові ваги (якщо є)


    # Ініціалізація параметрів (середніх значень, ковариацій та ваг)
    def initialize_params(self, X):
        n_samples, n_features = X.shape  # Отримуємо розміри вибірки
        np.random.seed(self.random_state)  # Встановлюємо фіксований стан для відтворюваності

        # Якщо задані користувацькі ініціалізації — використовуємо їх
        if self.means_init is not None:
            self.means = np.array(self.means_init)
        else:
            if self.init_params == 'random':
                self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]  # Випадкові центри

            elif self.init_params == 'kmeans':
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=self.n_components, n_init=10, random_state=self.random_state)
                kmeans.fit(X)
                self.means = kmeans.cluster_centers_  # Центри кластерів з k-means

            elif self.init_params == 'k-means++':
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=self.n_components, init='k-means++', n_init=1, random_state=self.random_state)
                kmeans.fit(X)
                self.means = kmeans.cluster_centers_

            else:
                raise ValueError(f"Невідомий метод ініціалізації: {self.init_params}")

        # Встановлення початкових ваг
        if self.weights_init is not None:
            self.weights = np.array(self.weights_init)
        else:
            self.weights = np.ones(self.n_components) / self.n_components  # Рівні ваги

        # Встановлення ковариаційних матриць 
        if self.precisions_init is not None:
            self.covariances = np.array(self.precisions_init)
        else:
            if self.covariance_type == 'full':
                # Окрема повна ковариаційна матриця для кожної компоненти
                self.covariances = np.array([
                    np.cov(X.T) + self.reg_cov * np.eye(n_features)
                    for _ in range(self.n_components)
                ])
            elif self.covariance_type == 'diag':
                # Окрема діагональна ковариація (дисперсія по кожній ознаці)
                var = np.var(X, axis=0) + self.reg_cov
                self.covariances = np.array([
                    np.diag(var)
                    for _ in range(self.n_components)
                ])
            elif self.covariance_type == 'spherical':
                # Однакове значення дисперсії для всіх ознак у кожному компоненті
                var = np.var(X) + self.reg_cov
                self.covariances = np.array([
                    np.eye(n_features) * var
                    for _ in range(self.n_components)
                ])
            elif self.covariance_type == 'tied':
                # Одна загальна ковариаційна матриця для всіх компонентів
                tied_cov = np.cov(X.T) + self.reg_cov * np.eye(n_features)
                self.covariances = np.array([tied_cov for _ in range(self.n_components)])
            else:
                raise NotImplementedError(f"Тип ковариацій '{self.covariance_type}' поки не підтримується.")


        # Ініціалізація матриці відповідальностей
        self.resp = np.zeros((n_samples, self.n_components))  # Порожня матриця відповідальностей

        # n_samples, n_features = X.shape  # Отримуємо розміри вибірки
        # np.random.seed(self.random_state)  # Фіксування випадкового стану для відтворюваності результатів
        # self.means = X[np.random.choice(n_samples, self.n_components, replace=False)]  # Вибір початкових середніх значень
        # self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])  # Ініціалізація ковариацій (ідентичні матриці)
        # self.weights = np.ones(self.n_components) / self.n_components  # Рівні ваги для кожного компонента
        # self.resp = np.zeros((n_samples, self.n_components))  # Матриця відповідальностей (яка компонента є більш ймовірною)

    # Функція для обчислення ймовірності за багатовимірним гауссівським розподілом
    def gaussian_pdf(self, X, mean, cov):
        n = X.shape[1]  # Кількість ознак (розмірність простору)
        diff = X - mean  # Вектор відхилення кожної точки від середнього значення

        if self.covariance_type == 'full' or self.covariance_type == 'tied':
            # У випадку повної або загальної (спільної) ковариаційної матриці
            try:
                inv_cov = np.linalg.inv(cov)  # Обернена матриця ковариацій
                det_cov = np.linalg.det(cov)  # Визначник ковариаційної матриці
            except np.linalg.LinAlgError:
                # Якщо матриця вироджена або її не вдається інвертувати — повертаємо нулі
                return np.zeros(X.shape[0])

            # Обчислення експоненційного компонента формули густини
            exp_term = np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=1))
            # Нормалізуючий множник у знаменнику
            denom = np.sqrt((2 * np.pi) ** n * det_cov)
            return exp_term / denom

        elif self.covariance_type == 'diag':
            # Для діагональної ковариації (відсутність кореляцій між ознаками)
            var = np.diag(cov)  # Діагональні значення — дисперсії по кожній ознаці
            exp_term = np.exp(-0.5 * np.sum((diff ** 2) / var, axis=1))  # Поелементне ділення
            denom = np.sqrt((2 * np.pi) ** n * np.prod(var))  # Добуток всіх дисперсій у знаменнику
            return exp_term / denom

        elif self.covariance_type == 'spherical':
            # Для сферичної ковариації (однакова дисперсія для всіх ознак)
            var = cov[0, 0]  # Єдина дисперсія, взята з елемента матриці
            exp_term = np.exp(-0.5 * np.sum((diff ** 2), axis=1) / var)
            denom = np.sqrt((2 * np.pi * var) ** n)
            return exp_term / denom

        else:
            # Якщо тип ковариації не підтримується
            raise NotImplementedError(f"Тип ковариацій '{self.covariance_type}' поки не підтримується.")



    # E-етап (очікування): обчислення відповідальностей (ймовірностей належності кожної точки до кожного кластеру)
    def e_step(self, X):
        for i in range(self.n_components):
            # Якщо використовується "загальна" (tied) ковариація — для всіх компонент одна й та ж матриця
            if self.covariance_type == 'tied':
                cov = self.covariances[0]
            else:
                cov = self.covariances[i]
            # Обчислення ймовірності належності кожної точки до компонента i (з урахуванням ваги компонента)
            self.resp[:, i] = self.weights[i] * self.gaussian_pdf(X, self.means[i], cov)
        
        # Нормалізація відповідальностей так, щоб сума по всім компонентам для кожної точки дорівнювала 1
        self.resp /= self.resp.sum(axis=1, keepdims=True)

        # Перевірка на наявність NaN (може свідчити про числову нестабільність)
        if np.any(np.isnan(self.resp)):
            raise ValueError("Знайдено NaN значення в матриці відповідальностей.")

   # M-етап (максимізація): оновлення параметрів моделі на основі відповідальностей
    def m_step(self, X):
        Nk = self.resp.sum(axis=0)  # Ефективна кількість точок, що належать кожному компоненту
        self.weights = Nk / X.shape[0]  # Оновлення ваг компонентів (як частка точок)
        self.means = (self.resp.T @ X) / Nk[:, np.newaxis]  # Оновлення середніх значень кожного компонента

        if self.covariance_type == 'tied':
            # Якщо всі компоненти мають спільну ковариаційну матрицю
            cov = np.zeros((X.shape[1], X.shape[1]))  # Початкова матриця ковариацій
            for i in range(self.n_components):
                diff = X - self.means[i]  # Відхилення від середнього
                # Зважене накопичення ковариаційної матриці для всіх компонентів
                cov += (self.resp[:, i][:, np.newaxis] * diff).T @ diff
            cov /= X.shape[0]  # Усереднення
            cov += self.reg_cov * np.eye(X.shape[1])  # Регуляризація — додаємо до діагоналі
            self.covariances = np.array([cov])  # Однакове значення для всіх компонентів

        else:
            # Індивідуальні ковариації для кожного компонента
            for i in range(self.n_components):
                diff = X - self.means[i]  # Відхилення від оновленого середнього

                if self.covariance_type == 'full':
                    # Повна матриця ковариацій (усі зв’язки між ознаками)
                    cov = (self.resp[:, i][:, np.newaxis] * diff).T @ diff / Nk[i]
                    cov += self.reg_cov * np.eye(X.shape[1])

                elif self.covariance_type == 'diag':
                    # Діагональна матриця — кожна ознака має свою дисперсію, без зв’язків
                    var = np.sum(self.resp[:, i][:, np.newaxis] * (diff ** 2), axis=0) / Nk[i]
                    cov = np.diag(var + self.reg_cov)

                elif self.covariance_type == 'spherical':
                    # Сферична матриця — однакова дисперсія для всіх ознак
                    var = np.sum(self.resp[:, i] * np.sum(diff ** 2, axis=1)) / (Nk[i] * X.shape[1])
                    cov = np.eye(X.shape[1]) * (var + self.reg_cov)

                else:
                    raise NotImplementedError(f"Тип ковариацій '{self.covariance_type}' поки не підтримується.")

                # Збереження оновленої ковариаційної матриці
                self.covariances[i] = cov

                # Перевірка на NaN у матриці
                if np.any(np.isnan(cov)):
                    raise ValueError(f"Знайдено NaN значення в матриці ковариацій для компонента {i}.")

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



# новий файл = n  в сек 
# сохраняю модель
# роблю предікт

#екран з моделями (скачування моделей)
#запуск моделей з різними моделями(готовими і ні)
#beta-flyer
#rocket launh data imu