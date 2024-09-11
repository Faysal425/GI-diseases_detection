################# feature map visualization PSE-CNN ####################

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model

# Load the model
model = load_model(r"D:\MN\gastro three stage\SaveFileForStage2\updated\PSECNN.h5")

# Confirm the model structure
model.summary()

# Function to load and preprocess image
def load_and_preprocess_image(img_path, target_size=(124, 124)):
    img = image.load_img(img_path, target_size=target_size, color_mode='rgb')  # Ensure RGB
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

img_path = r"samples.jpg"
sample_input = load_and_preprocess_image(img_path)

# Function to get the output of the last convolutional layer
def get_last_conv_layer_output(model):
    # Get the name of the last convolutional layer
    conv_layer_name = 'activation_5'
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer(conv_layer_name).output)
    return intermediate_model

# Create model to get outputs from the last convolutional layer
last_conv_model = get_last_conv_layer_output(model)

# Function to plot and save feature maps
def plot_feature_maps(activation, layer_name, num_columns=8, dpi=600, save_path='D:/MN/GastroVision2/Gastrovision/Gastrovision/psecnn_activation_5_feature_map3.svg'):
    num_features = activation.shape[-1]
    size = activation.shape[1]
    num_rows = num_features // num_columns
    if num_features % num_columns != 0:
        num_rows += 1

    display_grid = np.zeros((num_rows * size, num_columns * size))

    for i in range(num_features):
        row = i // num_columns
        col = i % num_columns
        channel_image = activation[0, :, :, i]
        channel_image -= channel_image.mean()
        channel_image /= channel_image.std()
        channel_image *= 64
        channel_image += 128
        channel_image = np.clip(channel_image, 0, 255).astype('uint8')
        display_grid[row * size: (row + 1) * size, col * size: (col + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]), dpi=dpi)
    # plt.title(f'Feature Maps of {layer_name}')
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.axis('off')
    plt.savefig(save_path, format='svg', dpi=dpi)
    plt.show()
    # plt.close()

# Get activations from the last convolutional layer
activations = last_conv_model.predict(sample_input)

# Plot and save the feature maps
plot_feature_maps(activations, 'activation_5')
