from flask import Flask, render_template, request, jsonify
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import os
from tempfile import NamedTemporaryFile

app = Flask(__name__)
model = keras.models.load_model('inception.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['image']
    
    # Create a temporary file to save the uploaded image
    temp_file = NamedTemporaryFile(delete=False)
    temp_file_path = temp_file.name
    file.save(temp_file_path)
    
    # Load and preprocess the uploaded image
    img = image.load_img(temp_file_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    
    # Close the temporary file
    temp_file.close()
    
    # Make predictions
    preds = model.predict(img)
    
    # Delete the temporary file
    os.unlink(temp_file_path)
    
    # Determine prediction result
    if preds[0][0] > 0.5:
        result = {"prediction": "The input image contains a kidney stone."}
    else:
        result = {"prediction": "The input image contains a normal kidney."}
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
