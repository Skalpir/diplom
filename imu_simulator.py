#file imu_simulator.py

#сохраняю параметры, делаю предикт, рисую графики, обновляю страницу что бы у пользователя подгрузились картинки,
#загружаю веса, получаю новые точки, дообучаю модель
#и так по кругу

#имитация работы модели в реальном времени

import pandas as pd
import time
import os

#все коментарі мають бути на українській мові
class IMUSimulator:
    def __init__(self, input_file, temp_file, global_file, model, batch_size=5, delay=0.5):
        self.input_file = input_file
        self.temp_file = temp_file
        self.global_file = global_file
        self.model = model
        self.batch_size = batch_size
        self.delay = delay
        self.position = 0  # каунтер СТРОКИ для позиції в даних

        # Читаємо дані з CSV файлу
        self.data = pd.read_csv(self.input_file)

        # Пусті 3 файли для тимчасового та глобального зберігання
        if not os.path.exists(self.temp_file):
            pd.DataFrame().to_csv(self.temp_file, index=False)
        if not os.path.exists(self.global_file):
            pd.DataFrame().to_csv(self.global_file, index=False)

    def simulate(self):
        while self.position < len(self.data):
            # --- Читаємо порцію даних ---
            batch = self.data.iloc[self.position:self.position + self.batch_size]

            # Дозаписуємо дані в тимчасовий файл для донавчання моделі (по суті тут наш IMU)
            if os.path.getsize(self.temp_file) > 0:
                temp_df = pd.read_csv(self.temp_file)
                temp_df = pd.concat([temp_df, batch], ignore_index=True)
            else:
                temp_df = batch.copy()
            temp_df.to_csv(self.temp_file, index=False)

            # Довчавка моделі на нових даних
            self.model.partial_fit(batch.values)

            # --- Отримуємо log_likelihood для batch-а(порції данних) ---
            log_likelihood = self.model.score_samples(batch.values)
            batch_with_logp = batch.copy()
            batch_with_logp["log_likelihood"] = log_likelihood

            # --- Дозаписуємо в глобальний файл цієї сессії (live_data.csv) ---
            if os.path.getsize(self.global_file) > 0:
                global_df = pd.read_csv(self.global_file)
                global_df = pd.concat([global_df, batch_with_logp], ignore_index=True)
            else:
                global_df = batch_with_logp
            global_df.to_csv(self.global_file, index=False)

            # --- Наступна порція ---
            self.position += self.batch_size
            time.sleep(self.delay)
