from flask import Flask, jsonify
from functions import *

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/welcome', methods=['GET'])
def welcome():
    return 'Система відстеження помилок у датчиках'

@app.route('/sensor-data', methods=['GET'])
def sensor_data():
    return jsonify(get_sensor_data())

if __name__ == "__main__":
    app.run(debug=True)
