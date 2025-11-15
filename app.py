import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

# --- Configuration ---

# Define the expected image size from your training script
IMG_SIZE = 96

# Define paths
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# --- Create Flask App ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER

# Ensure the upload and model folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# --- Helper Functions ---

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_available_models():
    """Scans the models folder and returns a list of .h5 models."""
    models = [f for f in os.listdir(app.config['MODEL_FOLDER']) if f.endswith('.h5')]
    return models

def preprocess_image(image_path):
    """
    Loads an image, resizes it to IMG_SIZE, rescales it (1./255),
    and expands its dimensions to match the model's expected input.
    """
    # Load the image using PIL (re-opening for safety)
    img = Image.open(image_path).convert('RGB')
    
    # 1. Resize the image (matching tf.keras.layers.Resizing)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    
    # Convert image to numpy array
    img_array = np.asarray(img)
    
    # 2. Rescale the image (matching tf.keras.layers.Rescaling(1./255))
    img_array = img_array / 255.0
    
    # 3. Add a batch dimension (model expects [batch_size, height, width, channels])
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch

# --- App Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main page with the model selection dropdown."""
    models = get_available_models()
    return render_template('index.html', models=models)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the image upload and prediction."""
    models = get_available_models() # Get models for re-render if error
    
    # --- 1. Get Form Data ---
    selected_model = request.form.get('model')
    
    if 'image' not in request.files:
        return render_template('index.html', models=models, error="No image file selected.")
    
    file = request.files['image']
    
    if file.filename == '':
        return render_template('index.html', models=models, error="No image file selected.")

    if not selected_model:
        return render_template('index.html', models=models, error="No model selected.")

    # --- 2. Save and Preprocess Image ---
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)
        
        try:
            # Preprocess the image
            image_batch = preprocess_image(image_path)
            
            # --- 3. Load Model and Predict ---
            model_path = os.path.join(app.config['MODEL_FOLDER'], selected_model)
            model = tf.keras.models.load_model(model_path)
            
            # Make prediction
            probabilities = model.predict(image_batch)
            
            # Get the class index with the highest probability
            predicted_class_index = np.argmax(probabilities[0])
            confidence = np.max(probabilities[0])
            
            # --- vvv THIS IS THE NEW PART vvv ---
            
            # 4. Map Index to Class Name
            #
            # IMPORTANT: Replace this list with your *actual* class names
            # The order MUST be the same as your model's training data.
            # (e.g., if 'chair' was 0, 'sofa' was 1, etc.)
            
            CLASS_NAMES = [
                'bed', 
                'chair', 
                'couch', 
                'table', 
                'wardrobe'
                # ... add all your classes here
            ]

            # 5. Create the output text
            try:
                # Look up the class name from the index
                predicted_class_name = CLASS_NAMES[predicted_class_index]
                
                # Format the text (e.g., 'night_stand' -> 'Night Stand')
                prediction_text = f"Predicted Class: {predicted_class_name.replace('_', ' ').title()}"
                confidence_text = f"Confidence: {confidence * 100:.2f}%"

            except IndexError:
                # This is a safety check in case the index is out of bounds
                prediction_text = f"Predicted Index: {predicted_class_index} (Error: CLASS_NAMES list is not up to date)"
                confidence_text = f"Confidence: {confidence * 100:.2f}%"

            # --- ^^^ END OF NEW PART ^^^ ---

            return render_template('index.html', 
                                   models=models, 
                                   selected_model=selected_model,
                                   prediction=prediction_text,  # <-- This now contains the name
                                   confidence=confidence_text,
                                   image_file=filename)

        except Exception as e:
            # Show the new error you found, or any other error
            return render_template('index.html', models=models, error=f"An error occurred: {e}")
            
    else:
        return render_template('index.html', models=models, error="Invalid file type. Please upload an image.")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serves the uploaded image to display it on the page."""
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# --- Run the App ---
if __name__ == '__main__':
    # Make sure static/uploads directory exists for serving images
    os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
    # Move the 'uploads' folder (where files are saved) inside 'static'
    # This allows Flask to serve them.
    app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
    
    app.run(debug=True) # Set debug=False for production
    
# End of app.py