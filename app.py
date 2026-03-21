from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from googletrans import Translator
import numpy as np
import os

from cures import cure_dict  # cure_dict is in cures.py

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model = load_model('best_model.keras')

# Image input size
IMG_SIZE = (224, 224)

# Class labels (must match training order)
class_names = ['Apple scab',
 'Apple Black rot',
 'Apple Cedar apple rust',
 'Apple healthy',
 'Blueberry healthy',
 'Cherry including sour Powdery mildew',
 'Cherry including sour healthy',
 'Corn maize Cercospora leaf spot Gray leaf spot',
 'Corn maize Common rust',
 'Corn maize Northern Leaf Blight',
 'Corn maize healthy',
 'Grape Black_rot',
 'Grape Esca Black Measles',
 'Grape Leaf blight Isariopsis Leaf Spot',
 'Grape healthy',
 'Orange Haunglongbing Citrus greening',
 'Peach Bacterial spot',
 'Peach healthy',
 'Pepper bell Bacterial spot',
 'Pepper bell healthy',
 'Potato Early blight',
 'Potato Late blight',
 'Potato healthy',
 'Raspberry healthy',
 'Soybean healthy',
 'Squash Powdery mildew',
 'Strawberry Leaf scorch',
 'Strawberry healthy',
 'Tomato Bacterial spot',
 'Tomato Early blight',
 'Tomato Late blight',
 'Tomato Leaf Mold',
 'Tomato Septoria leaf spot',
 'Tomato Spider mites Two spotted spider mite',
 'Tomato Target Spot',
 'Tomato Tomato Yellow Leaf Curl Virus',
 'Tomato Tomato mosaic virus',
 'Tomato healthy']

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE, color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'})

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    img_array = prepare_image(filepath)
    predictions = model.predict(img_array)
    predicted_class_raw = class_names[np.argmax(predictions[0])]  # e.g., Grape___Leaf_blight_(Isariopsis_Leaf_Spot)

    # Fix formatting for lookup and translation
    predicted_key = predicted_class_raw.replace('___', '_')
    treatment = cure_dict.get(predicted_key, "No treatment info available.")

    # Translation support
    lang = request.form.get('lang')
    readable_class = predicted_class_raw.replace('___', ' ').replace('_', ' ')
    translated_class = readable_class
    translated_treatment = treatment

    if lang:
        try:
            translator = Translator()
            translated_class = translator.translate(readable_class, dest=lang).text
            translated_treatment = translator.translate(treatment, dest=lang).text
        except Exception as e:
            print("Translation error:", e)

    return jsonify({
        'class': translated_class,
        'treatment': translated_treatment,
        'image_url': filepath
    })

if __name__ == '__main__':
    # التأكد من وجود مجلد الرفع عشان ما يعطيك Error لما يحفظ الصورة
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)