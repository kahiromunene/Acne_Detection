from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained CNN model
model = load_model('acne_model.tflite')

# Define target size for image resizing based on the model's input size
img_size = (224, 224)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image array
    return img_array

@app.route('/')
def index():
    return render_template('analysis.html')

@app.route('/predict', methods=['POST'])
def predictions():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        img_array = preprocess_image(file)
        prediction = model.predict(img_array)
        # Assuming your model output is binary (0 for no acne, 1 for acne)
        result = 'Acne Detected' if prediction[0][0] > 0.5 else 'No Acne Detected'
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
