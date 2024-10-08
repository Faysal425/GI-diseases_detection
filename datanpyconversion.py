import os
import cv2
import numpy as np

data_path = r'D:\MN\gastro three stage\stage1'

categories = os.listdir(data_path)
print("Number of classes:", categories)
noofClasses = len(categories)
print("Total number of classes:", noofClasses)
print("Importing images")

labels = [i for i in range(len(categories))]

label_dict = dict(zip(categories, labels))  # empty dictionary

print("Label dictionary:", label_dict)
print("Categories:", categories)
print("Labels:", labels)

img_size = 124
data = []
target = []

for category in categories:
    folder_path = os.path.join(data_path, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        try:
            # Convert the image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize the grayscale image to 124x124
            imga = cv2.resize(gray_img, (img_size, img_size))

            # Apply Z-score normalization to the resized image
            mean = np.mean(imga)
            std = np.std(imga)
            imga = (imga - mean) / (std if std != 0 else 1)  # Avoid division by zero

            data.append(imga)
            target.append(label_dict[category])
        except Exception as e:
            print(f'Exception processing image {img_name}: {e}')

data = np.array(data)
target = np.array(target)

print(data.shape)
print(target.shape)

# Save the processed data and labels
np.save("D:/MN/gastro three stage/SaveFileForStage1/FirstStageX.npy", data)
np.save("D:/MN/gastro three stage/SaveFileForStage1/SecondStageX.npy", target)

####### image shape checking ##########
import numpy as np

# Load the numpy array containing the images
images_rgb = np.load(r"D:/MN/gastro three stage/SaveFileForStage1/FirstStageX.npy")

# Check the shape of the numpy array
print("Shape of images after conversion to RGB:", images_rgb.shape)
