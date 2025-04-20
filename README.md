# Frost Detection on Martian Terrain Using Deep Learning

## Overview

This project focuses on detecting frost in high-resolution Martian terrain images obtained from NASA's HiRISE (High Resolution Imaging Science Experiment) dataset. The goal is to build image classifiers using both a custom Convolutional Neural Network (CNN + MLP) and transfer learning techniques with pre-trained models to accurately classify image tiles as either **frost** or **background**.

## Dataset

- **Source**: [NASA JPL Dataverse](https://dataverse.jpl.nasa.gov/dataset.xhtml?persistentId=doi:10.48577/jpl.QJ9PYA)
- **Images**: 299x299 pixel tiles generated from larger HiRISE subframes.
- **Classes**: Binary classification — `frost` or `background`.
- **Total tiles**: ~119,920

Data was split into training, validation, and test sets using provided `train/source/images.txt`, `test/source/images.txt`, and `val/source/images.txt` files.

## Project Structure

### 1. Data Preprocessing
- Augmentation techniques applied: rotation, flip, zoom, shift, contrast adjustment.
- Normalization of pixel values.
- Image generators created for training, validation, and test datasets.

### 2. CNN + MLP Model
- 3 Convolutional layers with ReLU activation, batch normalization, max-pooling.
- 1 Dense MLP layer with dropout and L2 regularization.
- Trained for 20 epochs using early stopping based on validation loss.
- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  
- Evaluated using Precision, Recall, and F1 Score.

### 3. Transfer Learning
- Pre-trained models used:
  - `EfficientNetB0`
  - `ResNet50`
  - `VGG16`
- All convolutional layers frozen; only classifier layers trained.
- Custom top layers with dropout, batch normalization, and ReLU + softmax output.
- Trained for 20 epochs using early stopping.

## Results

| Model          | Accuracy | Precision | Recall | F1 Score |
|----------------|----------|-----------|--------|----------|
| CNN + MLP      | ~94.4%   | 0.945     | 0.944  | 0.944    |
| VGG16          | ~97.4%   | 0.974     | 0.974  | 0.974    |
| ResNet50       | ~97.9%   | 0.979     | 0.979  | 0.979    |
| EfficientNetB0 | ~98.1%   | 0.981     | 0.981  | 0.981    |

## Visuals

- Plots of training vs validation loss and accuracy for all models.
- Bar chart comparing model performance metrics.

## Conclusion

Transfer learning significantly outperformed the custom CNN model. Among the transfer learning approaches, `EfficientNetB0` gave the highest performance with minimal training time and superior generalization.

## Requirements

- Python ≥ 3.7  
- TensorFlow ≥ 2.8  
- Keras ≥ 2.8  
- OpenCV  
- NumPy, Pandas, Scikit-learn, Matplotlib  

Install all dependencies with:

```bash
pip install -r requirements.txt
