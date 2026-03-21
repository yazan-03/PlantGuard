from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from googletrans import Translator
import numpy as np
import os

# استيراد قاموس العلاجات
try:
    from cures import cure_dict
except ImportError:
    cure_dict = {}
    print("Warning: cures.py not found or cure_dict missing.")

app = Flask(__name__)

# إعداد المجلدات - استخدام /tmp للسيرفرات لضمان صلاحيات الكتابة
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# التأكد من وجود المجلد عند تشغيل التطبيق (مهم جداً لـ Gunicorn)
try:
    # تحميل الموديل بدون Compile لتجنب تضارب الطبقات
    model = load_model('best_model.keras', compile=False)
    
    # إعادة بناء الموديل يدوياً بالأبعاد الصحيحة لـ EfficientNetB0
    model.build((None, 224, 224, 3)) 
    
    print("✅ Model loaded successfully using build() method!")
except Exception as e:
    model = None
    print(f"❌ Critical Error: {e}")

# حجم الصورة المطلوب للموديل
IMG_SIZE = (224, 224)

# أسماء الفئات
class_names = [
    'Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy',
    'Blueberry healthy', 'Cherry including sour Powdery mildew', 'Cherry including sour healthy',
    'Corn maize Cercospora leaf spot Gray leaf spot', 'Corn maize Common rust',
    'Corn maize Northern Leaf Blight', 'Corn maize healthy', 'Grape Black_rot',
    'Grape Esca Black Measles', 'Grape Leaf blight Isariopsis Leaf Spot', 'Grape healthy',
    'Orange Haunglongbing Citrus greening', 'Peach Bacterial spot', 'Peach healthy',
    'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight',
    'Potato Late blight', 'Potato healthy', 'Raspberry healthy', 'Soybean healthy',
    'Squash Powdery mildew', 'Strawberry Leaf scorch', 'Strawberry healthy',
    'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight',
    'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
    'Tomato Spider mites Two spotted spider mite', 'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus', 'Tomato Tomato mosaic virus', 'Tomato healthy'
]

def prepare_image(img_path):
    """تجهيز الصورة لـ EfficientNetB0 (بدون تقسيم على 255)"""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # ملاحظة: EfficientNetB0 تقوم بالـ Preprocessing داخلياً (Scaling)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # حفظ الصورة
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # التوقع
        img_array = prepare_image(filepath)
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions)
        predicted_class_raw = class_names[predicted_idx]

        # جلب العلاج من القاموس
        # تنظيف الاسم للبحث (تبديل الفراغات بـ _ حسب مفاتيح قاموسك)
        predicted_key = predicted_class_raw.replace(' ', '_').replace('___', '_')
        treatment = cure_dict.get(predicted_key, "No treatment info available for this condition.")

        # دعم الترجمة
        lang = request.form.get('lang')
        readable_class = predicted_class_raw.replace('_', ' ')
        
        translated_class = readable_class
        translated_treatment = treatment

        if lang and lang != 'en':
            try:
                translator = Translator()
                translated_class = translator.translate(readable_class, dest=lang).text
                translated_treatment = translator.translate(treatment, dest=lang).text
            except Exception as e:
                print(f"Translation error: {e}")

        return jsonify({
            'class': translated_class,
            'treatment': translated_treatment,
            'image_url': filepath
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    # تشغيل محلي (Local)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)