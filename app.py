import os
import csv
from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory, jsonify, send_file
from werkzeug.utils import secure_filename
from datetime import datetime
from sklearn_gmm import SklearnGMMAnalyzer
from my_gmm import MyGMMAnalyzer
from AnomalyAnalyzer import GMMAnomalyAnalyzer

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Потрібний для flash повідомлень
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv'}
RESULTS_FOLDER = os.path.join(os.getcwd(), 'results')
app.config.from_pyfile('config.py')

# 🌐 Головна сторінка
@app.route('/')
def index():
    return render_template('index.html')
# 🔧 Перевірка розширення файлу
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 📂 Завантаження файлів
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Файл не вибрано!')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Ім\'я файлу порожнє!')
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash('Дозволено лише файли з розширенням .csv')
            return redirect(request.url)
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        flash(f'Файл "{filename}" успішно збережено!')
        return redirect(url_for('upload'))

    # 🗂️ Список .csv файлів у папці uploads
    all_files = os.listdir(app.config['UPLOAD_FOLDER'])
    csv_files = [f for f in all_files if allowed_file(f)]

    return render_template('upload.html', files=csv_files)

# 📥 Завантаження файлу за ім'ям
@app.route('/uploads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

# 📊 Результати
@app.route('/results')
def results():
    folders = [f for f in os.listdir(RESULTS_FOLDER) if os.path.isdir(os.path.join(RESULTS_FOLDER, f))]
    return render_template('results.html', folders=folders)

# 📄 Сторінка перегляду одного конкретного результату
@app.route('/results/<folder_name>')
def result_detail(folder_name):
    folder_path = os.path.join(RESULTS_FOLDER, folder_name)
    if not os.path.exists(folder_path):
        return f"Папка '{folder_name}' не знайдена", 404

    files = os.listdir(folder_path)
    images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    texts = [f for f in files if f.lower().endswith(('.txt', '.log'))]
    csvs = [f for f in files if f.lower().endswith('.csv')]

    # Читання 
    text_contents = {}
    for txt in texts:
        
        try:
            with open(os.path.join(folder_path, txt), "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines() #[:50] # Обмежити до 50 рядків? Чи показувати всі?
                text_contents[txt] = ''.join(lines)
        except Exception as e:
            text_contents[txt] = f"[Помилка при читанні файлу: {e}]"

    csv_path = os.path.join(folder_path, "processed_data.csv")
    analyzer = GMMAnomalyAnalyzer()
    #TODO подумати чи потрібно дати можливість передавати threshold користувачу, якщо так то буде доречно винести це в окремий раут і зробити форму і кнопку.
    result = analyzer.run(csv_path, output_dir=folder_path, threshold=0.05)

    #print(text_contents)  # Для отладки
    return render_template(
        'result_detail.html',
        anomalies=result["anomalies"],
        folder=folder_name,
        images=images,
        texts=texts,
        csvs=csvs,
        text_contents=text_contents
    )

# @app.route('/results/<folder_name>/anomalies')
# def analyze_result_folder(folder_name):
#     # 📁 Повний шлях до папки з результатами
#     folder_path = os.path.join(RESULTS_FOLDER, folder_name)

#     if not os.path.exists(folder_path):
#         return jsonify({"success": False, "error": f"Папка '{folder_name}' не знайдена"}), 404

#     # 📄 Шлях до CSV з обробленими даними
#     csv_path = os.path.join(folder_path, "processed_data.csv")
#     if not os.path.exists(csv_path):
#         return jsonify({"success": False, "error": "Файл 'processed_data.csv' не знайдено"}), 404

#     try:
#         # 💡 Зчитуємо threshold з параметра запиту або беремо дефолтне значення
#         try:
#             threshold = float(request.args.get("threshold", 0.05))
#         except ValueError:
#             return jsonify({"success": False, "error": "Невірне значення threshold"}), 400

#         # 🧠 Створюємо аналізатор і запускаємо обробку
#         analyzer = GMMAnomalyAnalyzer()
#         result = analyzer.run(csv_path, output_dir=folder_path, threshold=threshold)

#         return jsonify({
#             "success": True,
#             "data": result
#         })

#     except Exception as e:
#         return jsonify({
#             "success": False,
#             "error": f"Помилка під час обробки: {str(e)}"
#         }), 500



# 🧾 Скачування будь-якого файлу з папки результатів
@app.route('/results/<folder_name>/<filename>')
def download_result_file(folder_name, filename):
    folder_path = os.path.join(RESULTS_FOLDER, folder_name)
    return send_from_directory(folder_path, filename)

# 🔮 GMM аналіз
@app.route('/gmm', methods=['GET', 'POST'])
def gmm():
    upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads')
    upload_folder_path = os.path.join(app.root_path, upload_folder)
    results_folder = os.path.join(app.root_path, 'results')
    
    files = []
    models = []

    try:
        for filename in os.listdir(upload_folder_path):
            if filename.endswith('.csv'):
                filepath = os.path.join(upload_folder_path, filename)
                stat = os.stat(filepath)
                files.append({
                    'name': filename,
                    'mtime': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                    'size': f'{stat.st_size / 1024:.1f} KB'
                })
    except Exception as e:
        flash(f"Не вдалося завантажити список файлів: {str(e)}")

    # results 
    try:
        if os.path.exists(results_folder):
            # print(results_folder)
            for model_name in os.listdir(results_folder):
                # print(model_name)
                model_dir = os.path.join(results_folder, model_name)
                # print(model_dir)
                if os.path.isdir(model_dir):
                    model_file = os.path.join(model_dir, 'gmm_params.json')
                    # print(model_file)
                    if os.path.exists(model_file):
                        print(model_file)
                        stat = os.stat(model_file)
                        models.append({
                            'name': model_name,
                            'path': model_file,
                            'mtime': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
                        })
    except Exception as e:
        flash(f"Не вдалося завантажити список моделей: {str(e)}")

    # print("models", models)
    return render_template('gmm.html', files=files, models=models)

# Завантаження моделі
@app.route('/download_gmm_model', methods=['POST'])
def download_gmm_model():
    model_path = request.form.get('model_path')

    if not model_path or not os.path.exists(model_path):
        flash("Файл не знайдено або шлях недійсний.")
        return redirect(url_for('gmm'))

    try:
        return send_file(model_path, as_attachment=True)
    except Exception as e:
        flash(f"Не вдалося завантажити файл: {str(e)}")
        return redirect(url_for('gmm'))


@app.route('/run_gmm', methods=['POST'])
def run_gmm():
    # Отримуємо параметри з форми
    n_components = int(request.form.get('n_components', 3))
    max_iter = int(request.form.get('max_iter', 100))  # Поки не використовується
    covariance_type = request.form.get('covariance_type', 'full')  # Поки не використовується
    gmm_impl = request.form.get('gmm_impl', 'sklearn-gmm')

    # Отримуємо файл
    uploaded_file = request.files.get('file')
    selected_file = request.form.get('selected_file')
    filename = None
    filepath = None

    # Завантажений файл або вибраний зі списку
    if uploaded_file and uploaded_file.filename != '':
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(filepath)
    elif selected_file:
        filename = secure_filename(selected_file)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    else:
        flash("Файл не обрано або не завантажено.")
        return redirect(url_for('gmm'))

    # Запускаємо GMM-аналіз
    print(gmm_impl)
    try:
        if gmm_impl == 'my-gmm':
            analyzer = MyGMMAnalyzer(
                csv_path=filepath,
                n_components=n_components,
            )
            print("my_gmm.py")
            analyzer.run()  # Запускаємо аналіз
            result_dir = analyzer.output_dir  # Папка з результатами

        elif gmm_impl == 'sklearn-gmm':
            analyzer = SklearnGMMAnalyzer(
                csv_path=filepath,
                n_components=n_components
            )
            print("sklearn_gmm.py")
            analyzer.run()  # Запускаємо аналіз
            result_dir = analyzer.output_dir  # Папка з результатами

    except Exception as e:
        flash(f"Помилка під час виконання GMM: {str(e)}")
        return redirect(url_for('gmm'))

    flash(f"GMM аналіз завершено для файлу: {filename}")
    return redirect(url_for('results', folder=result_dir))

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
