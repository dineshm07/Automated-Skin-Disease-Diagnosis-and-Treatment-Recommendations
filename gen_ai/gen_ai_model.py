import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import subprocess
import json
from gen_ai.grad_cam import get_img_array, make_gradcam_heatmap, save_and_display_gradcam

# === Constants ===
IMG_SIZE = (160, 160)
CLASS_NAMES = ['df', 'bcc', 'bkl', 'akiec', 'mel', 'nv', 'vasc']
MODEL_PATH = os.path.join('model/mobilenetv2 models', 'mobilenetv2_model.h5')
OUTPUT_DIR = 'gen_ai/gen_ai_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_image_and_generate_response(image_path):

    # === Load Model ===
    print("[INFO] Loading model...")
    model = load_model(MODEL_PATH, compile=False)
    print("[INFO] Model loaded successfully.")

    # === Preprocess and Predict ===
    print(f"[INFO] Processing image: {image_path}")
    img_array = get_img_array(image_path, size=IMG_SIZE)

    # Predict class
    preds = model.predict(img_array)
    pred_class_idx = np.argmax(preds[0])
    pred_class = CLASS_NAMES[pred_class_idx]
    confidence = preds[0][pred_class_idx] * 100
    print(f"[INFO] Predicted class: {pred_class} ({confidence:.2f}%)")

    # === Grad-CAM Visualization ===
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name='Conv_1')
    output_path = os.path.join(OUTPUT_DIR, f"gradcam_{os.path.basename(image_path)}")
    save_and_display_gradcam(image_path, heatmap, output_path=output_path)
    print(f"[INFO] Grad-CAM saved to {output_path}")

    # === Prepare Prompt for LLM (Ollama) ===
    gradcam_summary = "Red region is the dominant focus area in the lesion core."
    prompt_data = {
        "image_path": os.path.basename(output_path),
        "prediction": pred_class,
        "summary": gradcam_summary
    }

    prompt = f"""
    You are a medical assistant AI helping explain skin disease diagnoses.

    Based on the following information:
    - Predicted Disease: {prompt_data['prediction']}
    - Grad-CAM Summary: {prompt_data['summary']}

    Provide a clear and concise explanation in the following format:

    About the Disease:
    ...

    Causes:
    ...

    Remedies:
    ...

    Keep the content under 200 words total.
    """

    # === Call Ollama LLM ===
    print("[INFO] Generating explanation using Ollama...")
    try:
        result = subprocess.run([
            "ollama", "run", "llama3", prompt
        ], capture_output=True, text=True, check=True,encoding="utf-8",errors="replace")
        llm_response = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("[ERROR] Ollama failed to generate response.")
        llm_response = "[LLM ERROR] Could not generate explanation."

    # === Save LLM Output ===
    llm_output_path = os.path.join(OUTPUT_DIR, f"llm_output_{os.path.basename(image_path).split('.')[0]}.txt")
    with open(llm_output_path, 'w') as f:
        f.write(llm_response)

    print("[INFO] Explanation from LLM:")
    print(llm_response)
    print(f"[INFO] Saved to: {llm_output_path}")
    return {

        "pred_class": pred_class,
        "conf_score":round(confidence,2),
        "gradcam_path": output_path,
        "llm_output": llm_response
    }

    
