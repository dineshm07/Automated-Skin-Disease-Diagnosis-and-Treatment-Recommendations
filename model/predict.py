import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Prevent TensorFlow from allocating all GPU memory
def configure_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("[INFO] GPU memory growth enabled.")
        except Exception as e:
            print(f"[WARNING] Couldn't set memory growth: {e}")
    else:
        print("[INFO] No GPU found. Using CPU.")

# Clear any previous sessions
tf.keras.backend.clear_session()

# Configure GPU settings
configure_gpu()

# Set image size and class names
IMG_SIZE = (160, 160)
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Load the trained model
MODEL_PATH = './mobilenetv2 models/mobilenetv2_model.h5'
model = load_model(MODEL_PATH, compile=False)
print("[INFO] Model loaded successfully.")

# Function to predict a single image
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"[ERROR] Image path '{image_path}' does not exist.")
        return

    # Load and preprocess image
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)  # MobileNetV2 preprocessing
    img_array = np.expand_dims(img_array, axis=0)

    # Run prediction
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])
    pred_label = CLASS_NAMES[pred_index]
    confidence = preds[0][pred_index] * 100

    # Print result
    print(f"[RESULT] Predicted Class: {pred_label}")
    print(f"[RESULT] Confidence: {confidence:.2f}%")

# Example usage: python predict.py path_to_image.jpg
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python predict.py path_to_image.jpg")
    else:
        image_path = sys.argv[1]
        predict_image(image_path)
