# Traffic Sign Recognition with PyTorch

This repository contains a Python notebook for a Traffic Sign Recognition (TSR) project using PyTorch. The goal of this project is to train a deep learning model to recognize and classify traffic signs from the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

## Prerequisites

Before running the notebook, make sure you have the following dependencies installed:

- `torch` and `torchvision`
- `numpy`
- `opencv-python`
- `pandas`
- `tqdm`
- `PIL`
- `seaborn`
- `matplotlib`
- `scikit-learn`

You can install these dependencies using the following command:

```bash
pip install -r requirements.txt
```
## Dataset
The GTSRB dataset is used for training and testing the model. The dataset is downloaded and extracted automatically in the notebook. It consists of various traffic sign images belonging to different classes.

## Usage
1. Clone the repository:
```bash
git clone https://github.com/zaheerh4ck3r/traffic-sign-classification.git
cd traffic-sign-classification
```
2. Open and run the Jupyter notebook `transfer_learning_for_image_classifciation_using_torchvision.ipynb` using your preferred Python environment.
3. Follow the instructions in the notebook to train the model, evaluate its performance, and make predictions.

## Note
The notebook includes sections for data preprocessing, model training, evaluation, and prediction. It also provides visualizations of training history, confusion matrix, and sample predictions.

Feel free to customize the notebook for your specific use case and experiment with different hyperparameters, architectures, or datasets.

**Note:** The notebook uses a pre-trained ResNet34 model. You can explore other pre-trained models provided by PyTorch for better performance.


