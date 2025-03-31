from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import base64
import logging

app = Flask(__name__)
model = load_model('handwritten_digit_recognition_model.h5')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # Ensure model is compiled

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

from flask import Flask, render_template

@app.route('/')
def index():
    return render_template('index.html')  # This will look inside /templates


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Log the incoming request data for debugging
        app.logger.debug(f"Received data: {data}")

        image_data = data.get('image')
        image_type = data.get('type')

        # Decode the base64 image based on the image type
        if not image_data or not image_type:
            return jsonify({'error': 'Invalid input data'}), 400

        if image_type == 'canvas':
            image_data = image_data.split(',')[1]  # Remove the "data:image/png;base64," part
        elif image_type == 'file':
            image_data = image_data.split(',')[1]

        # Log the image type and confirm it's being processed correctly
        app.logger.debug(f"Image type: {image_type}")

        # Decode the base64 image
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        image = image.convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to the model input size
        image = np.array(image)
        
        # Log the image array to ensure it's being processed correctly
        app.logger.debug(f"Image array shape before reshaping: {image.shape}")
        
        # Ensure the input shape is correct
        image = image.reshape((1, 28, 28, 1)).astype('float32') / 255
        
        app.logger.debug(f"Image array shape after reshaping: {image.shape}")

        # Predict the digit
        prediction = model.predict(image)
        predicted_digit = np.argmax(prediction)

        # Log the prediction result
        app.logger.debug(f"Prediction: {prediction}, Predicted digit: {predicted_digit}")

        return jsonify({'prediction': int(predicted_digit)})
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
