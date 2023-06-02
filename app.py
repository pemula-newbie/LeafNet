# =[Modules dan Packages]========================
import tensorflow as tf
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_ngrok import run_with_ngrok
import cv2

# =[Variabel Global]=============================
app = Flask(__name__, static_url_path='/static')
model = None

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]
@app.route("/")
def beranda():
    return render_template('index.html')

@app.route("/upload")
def upload_page():
    return render_template('upload.html')

@app.route("/inner-page")
def inner_page():
    return render_template('inner-page.html')

# [Routing untuk API]
@app.route("/api/deteksi", methods=['POST'])
def apiDeteksi():
    # Load model yang telah ditraining
    global model
    if model is None:
        model = tf.keras.models.load_model('model_leafnet_tf.h5')
    
    if request.method == 'POST':
        # Menerima file gambar yang dikirim dari frontend
        file = request.files['file']
        
        # Simpan file gambar ke direktori temporary
        file_path = 'static/temp/temp.jpg'
        file.save(file_path)
        
        # Membaca dan memproses gambar dengan OpenCV
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        
        # Melakukan prediksi menggunakan model
        prediksi = model.predict(image)
        class_index = np.argmax(prediksi)
        
        # Daftar kelas
        classes = ['class_0', 'class_1', 'class_2', 'class_3']
        
        # Mengambil label prediksi
        hasil_prediksi = classes[class_index]
        
        # Mengubah label prediksi menjadi teks yang lebih deskriptif
        if hasil_prediksi == 'class_0':
            hasil_prediksi = 'Apple Black Rot'
        elif hasil_prediksi == 'class_1':
            hasil_prediksi = 'Apple Cedar Apple Rust'
        elif hasil_prediksi == 'class_2':
            hasil_prediksi = 'Apple Healthy'
        else:
            hasil_prediksi = 'Apple Scab'
        
        # Menghapus file gambar temporary
        os.remove(file_path)
        
        # Return hasil prediksi dengan format JSON
        return jsonify({
            "prediksi": hasil_prediksi
        })

# =[Main]========================================

if __name__ == '__main__':
    # Run Flask di localhost
    run_with_ngrok(app)
    app.run()
