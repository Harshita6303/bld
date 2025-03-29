from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import uuid  # For unique filenames

app = Flask(__name__)
model = load_model('Trained_Model.keras')

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if it doesn't exist

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img_file = request.files['image']
        
        # Generate a unique filename (e.g., "abc123.jpg")
        img_name = f"{uuid.uuid4()}.jpg"
        img_path = os.path.join(UPLOAD_FOLDER, img_name)
        
        # Save the uploaded file
        img_file.save(img_path)
        
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        predicted_label = chr(class_index + 65)  # Convert to A/B/C/D
        
        # Delete the temporary file
        os.remove(img_path)
        
        return render_template('result.html', predicted_label=predicted_label)
  
@app.route('/test-css')  # Add this route
def test_css():
    return send_from_directory('static', 'style2.css')


if __name__ == '__main__':
    app.run(debug=True)