import os
import csv
from flask import Flask, request, redirect, url_for, render_template, flash, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Потрібний для flash повідомлень
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv'}
RESULTS_FOLDER = os.path.join(os.getcwd(), 'results')

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

    # читаем первые 50 строк каждого текстового файла
    text_contents = {}
    for txt in texts:
        try:
            with open(os.path.join(folder_path, txt), "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()[:50]
                text_contents[txt] = ''.join(lines)
        except Exception as e:
            text_contents[txt] = f"[Помилка при читанні файлу: {e}]"

    return render_template(
        'result_detail.html',
        folder=folder_name,
        images=images,
        texts=texts,
        csvs=csvs,
        text_contents=text_contents
    )

# 🧾 Скачування будь-якого файлу з папки результатів
@app.route('/results/<folder_name>/<filename>')
def download_result_file(folder_name, filename):
    folder_path = os.path.join(RESULTS_FOLDER, folder_name)
    return send_from_directory(folder_path, filename)

# 🔮 GMM аналіз
@app.route('/gmm', methods=['GET', 'POST'])
def gmm():
    if request.method == 'POST':
        # Заглушка для GMM, просто показуємо, сторінку
        # TODO треба буде реалізувати функції для відображенян списку завантежених файлів
        flash('Аналіз GMM запущено! (поки що заглушка для кнопки)')
        return redirect(url_for('results'))
    return render_template('gmm.html')

@app.route('/run_gmm', methods=['POST'])
# TODO реалізувати функцію для запуску GMM з параметрами
def run_gmm():
    # Отримуємо параметри з форми
    n_components = int(request.form.get('n_components', 3))
    max_iter = int(request.form.get('max_iter', 100))
    covariance_type = request.form.get('covariance_type', 'full')
    gmm_impl = request.form.get('gmm_impl', 'my-gmm')

    # Отримуємо файл
    uploaded_file = request.files.get('file')
    selected_file = request.form.get('selected_file')
    filename = None
    filepath = None

    # Перевіряємо, чи файл завантажено або вибрано
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

    # Читаем CSV
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        flash(f"Помилка читання CSV: {str(e)}")
        return redirect(url_for('gmm'))

    #TODO Місцева реалізація GMM , поки що я його не зробив так що би з параметрами, тому просто залишу це тут
    try:
        if gmm_impl == 'my-gmm':
            result = my_gmm(data, n_components=n_components, max_iter=max_iter, covariance_type=covariance_type)
        else:
            result = sklearn_gmm(data, n_components=n_components, max_iter=max_iter, covariance_type=covariance_type)
        
        # Допустимо, що результат зберігається в файл, поки що не знаю як буде на справді.
        output_path = os.path.join('results', f"result_{filename}")
        result.to_csv(output_path, index=False)

    except Exception as e:
        flash(f"Помилка під час виконання GMM: {str(e)}")
        return redirect(url_for('gmm'))

    flash(f"GMM аналіз завершено для файлу: {filename}")
    return redirect(url_for('results'))

if __name__ == '__main__':
    app.run(debug=True)
