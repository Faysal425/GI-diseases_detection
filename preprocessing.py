import cv2
import numpy as np
import os
from skimage import exposure, img_as_ubyte
from skimage.filters import gaussian

def process_image(image):
    # Erosion
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(cv2.cvtColor(eroded, cv2.COLOR_BGR2GRAY))
    clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)

    # Sharpening
    kernel_sharpening = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])
    sharpened = cv2.filter2D(clahe_image, -1, kernel_sharpening)

    # Gaussian filter
    gaussian_filtered = gaussian(sharpened, sigma=1, multichannel=True)

    # Convert to 8-bit image
    final_image = img_as_ubyte(gaussian_filtered)

    return final_image

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read image
            image = cv2.imread(input_path)

            if image is not None:
                processed_image = process_image(image)

                # Save processed image
                cv2.imwrite(output_path, processed_image)
                print(f"Processed and saved: {output_path}")
            else:
                print(f"Error reading image: {input_path}")

# Example usage
input_folder = 'path/to/your/input/folder'
output_folder = 'path/to/your/output/folder'
process_folder(input_folder, output_folder)
