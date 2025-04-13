from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

# Імена сенсорів для демонстрації
sensor_names = ['Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Accel_X', 'Accel_Y', 'Accel_Z']

# Заглушкові дані для журналу подій
fake_sensor_data = [
    {'time': '12:01:01', 'sensor': 'Gyro_X', 'value': 4.2, 'type': 'Аномалія'},
    {'time': '12:01:02', 'sensor': 'Accel_Z', 'value': -20.5, 'type': 'Підозра'},
    {'time': '12:01:03', 'sensor': 'Gyro_Y', 'value': 0.2, 'type': 'Норма'}
]

@app.route('/')
def index():
    # Повертає основну HTML-сторінку з кнопками та візуалізаціями
    return render_template('index.html')

@app.route('/load-data', methods=['POST'])
def load_data():
    # Заглушка — тут має бути логіка завантаження даних з файлу або потоку
    return jsonify({'status': 'ok', 'message': 'Дані завантажено'})

@app.route('/run-analysis', methods=['POST'])
def run_analysis():
    # Заглушка — тут має бути логіка аналізу даних (наприклад GMM)
    return jsonify({'status': 'ok', 'message': 'Аналіз завершено'})

@app.route('/get-log', methods=['GET'])
def get_log():
    # Повертає журнал подій для відображення у фронтенді
    return jsonify(fake_sensor_data)

@app.route('/download-report', methods=['GET'])
def download_report():
    # Заглушка — тут можна буде реалізувати генерацію CSV/звіту
    return jsonify({'status': 'ok', 'message': 'Скачування буде доступне пізніше'})

if __name__ == '__main__':
    app.run(debug=True)