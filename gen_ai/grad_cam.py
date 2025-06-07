import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os

def get_img_array(img_path, size):
    """
    Load and preprocess an image.
    """
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a given image and model.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, output_path="cam_output.jpg", alpha=0.3):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image at path: {img_path}")
    if heatmap is None or not isinstance(heatmap, np.ndarray):
        raise ValueError("Heatmap is invalid or None.")

    print(f"[DEBUG] Heatmap shape: {heatmap.shape}, dtype: {heatmap.dtype}, min: {np.min(heatmap)}, max: {np.max(heatmap)}")

    # ✅ Convert heatmap to float32 for cv2.resize compatibility
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    superimposed_img = heatmap_colored * alpha + img
    cv2.imwrite(output_path, np.uint8(superimposed_img))



    


