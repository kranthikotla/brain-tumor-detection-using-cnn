from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Load the model
model = load_model("best_model.h5")

# Define the prediction function
def predict_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize the image to match the input shape of the model
    img = np.array(img) / 255.0    # Convert the image to numpy array and normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)[0][0]  # Make prediction
    return prediction

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join('static/uploads', f.filename)
        f.save(file_path)
        prediction = predict_image(file_path)
        if prediction < 0.5:
            result = "The MRI image is of BRAIN TUMOR"
        else:
            result = "The MRI image is of HEALTHY BRAIN"
        return render_template('index.html', prediction=result, image_file=f.filename)

if __name__ == "__main__":
    app.run(debug=True)
