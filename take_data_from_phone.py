import subprocess
import csv
import time
import re

def log(logMessage):
    """Логування в файл"""
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{logMessage}\n")

last_timestamps = {"Gyroscope": None, "Accelerometer": None}

# CSV-файл
with open("sensor_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Sensor", "X", "Y", "Z"])

    while True:
        # підключемся до сенсора
        adb_sensors_command = "adb shell dumpsys sensorservice"
        sensors_process = subprocess.Popen(adb_sensors_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        current_sensor = None 
        buffer = []  # адб показує остані 10 - 50 записів, а не дані в реальному часі, поки що так

        for line in sensors_process.stdout:
            line = line.strip()

            #  Accelerometer
            if "icm4x6xx Accelerometer" in line:
                current_sensor = "Accelerometer"
                buffer = []  
                continue

            #  Gyroscope
            elif "icm4x6xx Gyroscope" in line:
                current_sensor = "Gyroscope"
                buffer = [] 
                continue

            # Читання даних
            if current_sensor and len(buffer) < 10:
                buffer.append(line)
                if len(buffer) < 10:
                    continue  

            # Парсинг
            if len(buffer) == 10:
                for data_line in buffer:
                    match = re.search(r"(\d+) \(ts=(\d+\.\d+), wall=\d+:\d+:\d+\.\d+\) ([\d\-.]+), ([\d\-.]+), ([\d\-.]+)", data_line)
                    if match:
                        sensor_index = int(match.group(1))
                        timestamp = float(match.group(2))  # Час
                        x, y, z = float(match.group(3)), float(match.group(4)), float(match.group(5))  # Координати

                        # Захист від дублікатів (кожен раз адб дає по останіх 10-50 записів і вони йдуть рандомно, треба якось боротись з дублікатами на етапі запису )
                        if last_timestamps[current_sensor] is None or timestamp > last_timestamps[current_sensor]:
                            writer.writerow([timestamp, current_sensor, x, y, z])
                            file.flush()
                            print(f"{timestamp}: {current_sensor} - {x}, {y}, {z}")  # дебаг

                            last_timestamps[current_sensor] = timestamp 

                # дроп буфера
                buffer = []
                current_sensor = None

        # Треба пооксперементувати, як швидко оновлються остані 10-50 записів в адб, можливо, треба зробити більше(3-5)
        time.sleep(0.5)
