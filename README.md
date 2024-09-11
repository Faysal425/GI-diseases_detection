# Image Processing and Model Training Pipeline

## Overview

This project develops a computer-aided diagnosis (CAD) system for gastrointestinal (GI) disorders using a three-stage deep learning framework. The architecture integrates a feature extractor, a Parallel Squeeze-and-Excitation Convolutional Neural Network (PSE-CNN), and Principal Component Analysis (PCA) with a Deep Extreme Learning Machine (DELM) classifier.

## Repository Structure

Here's a brief overview of the key files in the repository:

- `Installation_requirements.txt`: Lists the Python packages required for the project.
- `preprocessing.py`: Contains the image processing functions including erosion, CLAHE, sharpening, and Gaussian filtering.
- `kfold.py`: Implements 5-fold cross-validation to evaluate model performance on different subsets of the data.
- `train.py`: Trains various models, including PSE-CNN (proposed), Pr-CNN, DenseNet201, InceptionResNetV2, Xception, MobileNetV2, VGG16, and ResNet152V2.
- `test.py`: Evaluates all trained feature extraction models using PCA and the proposed Deep Extreme Learning Machine (DELM).

## Installation

To set up the project, you need to install the necessary Python packages. You can do this using `pip`. Follow the steps below:

1. Clone the repository to your local machine:
    ```python
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. Install the required packages:
    ```python
    pip install -r Installation_requirements.txt
    ```

## Usage

### Dataset Description
GastroVision is a multi-center open-access GI endoscopy dataset featuring a diverse range of anatomical landmarks, pathological abnormalities, polyp removal cases, and normal findings from the GI tract. The dataset includes 8,000 images across 27 distinct classes, collected from Baerum Hospital in Norway and Karolinska University in Sweden, with annotations and verifications performed by experienced GI endoscopists. The significance of the dataset is validated through extensive benchmarking with popular deep learning baseline models. GastroVision serves as a valuable resource for developing AI-based algorithms for GI disease detection and classification. The dataset is publicly available you can download it from [Dataset Paper](https://doi.org/10.1007/978-3-031-47679-2_10) or
[Kaggle](https://www.kaggle.com/datasets/debeshjha1/gastrovision) or [OSF Data Portal](https://osf.io/84e7f/) or [Google Drive](https://drive.google.com/drive/folders/1T35gqO7jIKNxC-gVA2YVOMdsL7PSqeAa?usp=sharing).

### Image Processing

The core functionality for image preprocessing is in `preprocessing.py`. The `process_image` function performs the following operations:

1. **Erosion**: Applies a morphological operation to reduce noise.
2. **CLAHE**: Enhances contrast in the image.
3. **Sharpening**: Applies a sharpening filter to enhance details.
4. **Gaussian Filter**: Applies a Gaussian filter to smooth the image.

```python
python preprocessing.py
```

### Cross-Validation

The `kfold.py` code performs k-fold cross-validation with an 80:10:10 split for training, validation, and test sets, dividing the dataset into `k` folds where each fold serves as a validation set once. It maintains the specified ratio across the folds and organizes data into structured directories for each subset. Optional randomization ensures unbiased distribution, and images are resized and saved in the specified format, facilitating comprehensive model evaluation and enhancing result generalizability.

```bash 
python kfold.py
```

### NPY conversion

The code converts images into `.npy` format, which is a binary file format used by NumPy to store arrays efficiently. It reads images from the specified source directory, processes them by resizing and converting them into arrays, and then saves these arrays in `.npy` format to the destination directory. This approach standardizes the data, reduces loading times, and streamlines the use of image data for machine learning tasks by making it compatible with NumPy-based workflows.

```python 
python datanpyconversion.py
```
For checking the shape of each `.npy` run this code:
```python 
import numpy as np

# Load the numpy array containing the images
images_rgb = np.load(r"D:/MN/gastro three stage/SaveFileForStage1/FirstStageX.npy")

# Check the shape of the numpy array
print("Shape of images after conversion to RGB:", images_rgb.shape)

```

### Model Training

The `train.py` script allows you to train several models, including:

- **PSE-CNN (proposed)**: A proposed model for image classification.
- **Pr-CNN**: A pre-existing CNN model.
- **DenseNet201**: A DenseNet architecture with 201 layers.
- **InceptionResNetV2**: An Inception network with residual connections.
- **Xception**: A deep convolutional network with depthwise separable convolutions.
- **MobileNetV2**: A lightweight mobile network architecture.
- **VGG16**: A classic deep learning model with 16 layers.
- **ResNet152V2**: A ResNet model with 152 layers and version 2 improvements.
you can also train this for 5-Fold cross validation dataset.

For each stage training, use:
```python 
python train.py
```

### Evaluation

The `test.py` script is used to evaluate the performance of all trained feature extraction models. It uses PCA for dimensionality reduction and tests the models with the proposed Deep Extreme Learning Machine (DELM).

For each stage testing, use:
```python 
python test.py
```

### Model Interpritability 

For interpritability testing two approaches were followed:
- For visualize the feature map of the convolutional layer.
- For model's predictive explainability (XAI) SHAP, Heatmap, GradCAM and Saliency map are visualized.  

For XAI, use:
```python 
python test.py
python XAI.py
```

## Contact Us

If you have any questions or need further information, feel free to reach out.

**Email:** [faysal.ahamed@ece.ruet.ac.bd](faysal.ahamed@ece.ruet.ac.bd)
