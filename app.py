from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('BrainTumorUpdatedClassifierModell(1).h5')

# Function to get class label
def get_class_label(prediction):
    classes = {0: 'glioma_tumor', 
               1: 'meningioma_tumor', 
               2: 'no_tumor', 
               3: 'pituitary_tumor'}
    return classes[prediction]

# Function to preprocess an image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from path: {image_path}")
        return None
    img = cv2.resize(image, (64, 64))  # Resize to match the model's input shape
    img = img / 255.0   # Normalize the image
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file
    file = request.files['file']
    # Save the file to disk
    file_path = 'uploads/' + file.filename
    file.save(file_path)
    
    # Preprocess image and predict class label
    input_img = preprocess_image(file_path)
    if input_img is not None:
        input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
        probabilities = model.predict(input_img)
        predicted_class_index = np.argmax(probabilities)
        predicted_class_label = get_class_label(predicted_class_index)
        
        if predicted_class_label == 'glioma_tumor':
            return "Sorry!!! you have glioma tumor. Gliomas can cause headaches, seizures, and difficulties with memory and concentration. They can be life-threatening depending on their location and aggressiveness. Please visit the doctor."
        elif predicted_class_label == 'meningioma_tumor':
            return "Sorry!!! you have meningioma tumor. Meningiomas may cause headaches, vision problems, seizures, and weakness in the limbs. Though usually benign, they can cause serious complications if they press on vital brain structures. Please visit the doctor."
        elif predicted_class_label == 'pituitary_tumor':
            return "Sorry!!! you have pituitary tumor. Pituitary tumors can lead to vision problems, headaches, hormonal imbalances, and even changes in mood and behavior. They require careful management to prevent serious health issues. Please visit the doctor."
        else:
            return "Good news!!! No tumor detected. Regular check-ups and a healthy lifestyle can help maintain your brain health."
    else:
        return "Error processing image"

if __name__ == '__main__':
    app.run(debug=True)
