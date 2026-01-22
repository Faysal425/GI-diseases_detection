## Overview

This project develops a computer-aided diagnosis (CAD) system for gastrointestinal (GI) disorders using a three-stage deep learning framework. The architecture integrates a feature extractor, a Parallel Squeeze-and-Excitation Convolutional Neural Network (PSE-CNN), and Principal Component Analysis (PCA) with a Deep Extreme Learning Machine (DELM) classifier.

## Installation

To set up the project, you need to install the necessary Python packages. You can do this using `pip`. Follow the steps below:

1. Clone the repository to your local machine:
    ```python
    git clone https://github.com/your-username/repo-name.git
    cd your-repo-name
    ```

2. Install the required packages:
    ```python
    pip install -r Installation_requirements.txt
    ```

## Usage

### Dataset Description
GastroVision is a multi-center open-access GI endoscopy dataset featuring a diverse range of anatomical landmarks, pathological abnormalities, polyp removal cases, and normal findings from the GI tract. The dataset includes 8,000 images across 27 distinct classes, collected from Baerum Hospital in Norway and Karolinska University in Sweden, with annotations and verifications performed by experienced GI endoscopists. The significance of the dataset is validated through extensive benchmarking with popular deep learning baseline models. GastroVision serves as a valuable resource for developing AI-based algorithms for GI disease detection and classification. The dataset is publicly available you can download it from [Dataset Paper](https://doi.org/10.1007/978-3-031-47679-2_10) or
[Kaggle](https://www.kaggle.com/datasets/debeshjha1/gastrovision) or [OSF Data Portal](https://osf.io/84e7f/).


### Citation

```bibtex
@article{AHAMED2025109503,
  title   = {Interpretable Deep Learning Architecture for Gastrointestinal Disease Detection: A Tri-Stage Approach with PCA and XAI},
  journal = {Computers in Biology and Medicine},
  volume  = {185},
  pages   = {109503},
  year    = {2025},
  issn    = {0010-4825},
  doi     = {10.1016/j.compbiomed.2024.109503}
}