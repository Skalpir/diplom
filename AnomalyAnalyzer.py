import pandas as pd
import matplotlib.pyplot as plt
import os

class GMMAnomalyAnalyzer:
    def __init__(self):
        # Нічого не ініціалізуємо тут — дані будуть приходити пізніше через .run()
        self.df = None
        self.anomaly_column = "Is_Anomaly"
        self.prob_column = "Log_Probability"
        self.timestamp_column = "Timestamp"

    def run(self, file_path, save_plots=True, output_dir="static/plots"):
        """
        Основний метод запуску аналізу. Приймає шлях до CSV-файлу та повертає результати аналізу.
        """
        self.df = pd.read_csv(file_path)

        # Підрахунок аномалій
        total_anomalies = self.df[self.anomaly_column].sum()
        top_anomalies = self.df[self.df[self.anomaly_column]].nlargest(10, self.prob_column).to_dict(orient="records")

        # Генерація графіків
        plot_paths = []
        if save_plots:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            path1 = os.path.join(output_dir, "log_probabilities.png")
            self._plot_log_probabilities(path1)
            plot_paths.append(path1)

            path2 = os.path.join(output_dir, "anomaly_hist.png")
            self._plot_anomaly_histogram(path2)
            plot_paths.append(path2)

        return {
            "total_anomalies": int(total_anomalies),
            "top_anomalies": top_anomalies,
            "plot_paths": plot_paths
        }

    def _plot_log_probabilities(self, path):
        # Побудова графіку логарифмічної ймовірності в часі
        plt.figure(figsize=(10, 4))
        plt.plot(self.df[self.timestamp_column], self.df[self.prob_column], label="Log Probability")
        plt.title("Log Probability over Time")
        plt.xlabel("Timestamp")
        plt.ylabel("Log Probability")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _plot_anomaly_histogram(self, path):
        # Побудова гістограми логарифмічної ймовірності
        plt.figure(figsize=(6, 4))
        self.df[self.prob_column].hist(bins=50)
        plt.title("Histogram of Log Probabilities")
        plt.xlabel("Log Probability")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
