import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython.display import Image, display
import tensorflow as tf
from tensorflow import keras

# Load the trained PSECNN model
model_path = 'PSECNN.h5'
model = keras.models.load_model(model_path)

# Define input size for PSECNN
img_size = (124, 124)

# Specify the last convolutional layer name
last_conv_layer_name = "Last_Convolution_Layer"

# Load and display the target image
img_path = 'image.jpg'
display(Image(img_path))

def get_img_array(img_path, size):
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Prepare image array and predict
img_array = get_img_array(img_path, size=img_size)
preds = model.predict(img_array)
print("Predicted:", preds)

# Generate and display Grad-CAM heatmap
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
plt.matshow(heatmap)
plt.show()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=1.00):
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    display(Image(cam_path))

save_and_display_gradcam(img_path, heatmap)

def heatmap_map(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    guided_grads = tf.cast(last_conv_layer_output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
    pooled_guided_grads = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
    guided_gradcam = last_conv_layer_output[0] @ pooled_guided_grads[..., tf.newaxis]
    guided_gradcam = tf.squeeze(guided_gradcam)
    guided_gradcam = tf.maximum(guided_gradcam, 0) / tf.math.reduce_max(guided_gradcam)
    return guided_gradcam.numpy()

# Generate and display Guided Grad-CAM heatmap
heatmap2 = heatmap_map(img_array, model, last_conv_layer_name)
def save_and_display_guided_gradcam(img_path, heatmap2, cam_path="cam.jpg", alpha=1.00):
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)
    heatmap2 = np.uint8(255 * heatmap2)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap2 = jet_colors[heatmap2]
    jet_heatmap2 = keras.preprocessing.image.array_to_img(jet_heatmap2)
    jet_heatmap2 = jet_heatmap2.resize((img.shape[1], img.shape[0]))
    jet_heatmap2 = keras.preprocessing.image.img_to_array(jet_heatmap2)
    superimposed_img = jet_heatmap2 * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    display(Image(cam_path))

save_and_display_guided_gradcam(img_path, heatmap2)

def generate_saliency_map(img_array, model, pred_index=None):
    img_tensor = tf.convert_to_tensor(img_array)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        pred_output = preds[:, pred_index]
    grads = tape.gradient(pred_output, img_tensor)
    guided_grads = tf.cast(img_tensor > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
    saliency_map = tf.reduce_max(guided_grads, axis=-1)
    saliency_map /= tf.reduce_max(saliency_map)
    return saliency_map.numpy()

# Generate and display Saliency map
saliency_map = generate_saliency_map(img_array, model)
saliency_map_2d = saliency_map[0]  # Ensure this matches the expected dimensions
plt.matshow(saliency_map_2d, cmap='viridis')
plt.show()
