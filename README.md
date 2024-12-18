# MNIST Handwritten Digits Recognition

## Project Overview

The MNIST Handwritten Digits Recognition project uses deep learning techniques to classify handwritten digits from the MNIST dataset. The model leverages a neural network architecture with dense layers and ReLU activation functions. The network is trained using TensorFlow and evaluated based on its accuracy in identifying digits from images.

The goal is to develop an accurate model capable of recognizing handwritten digits and to extend the model to handle custom images.

## Tools and Libraries Used

**Programming Language:**  
Python

**Libraries and Frameworks:**
- TensorFlow: For building, training, and evaluating the neural network model.
- OpenCV: For reading and processing images.
- NumPy: For numerical computations and data manipulation.
- Matplotlib: For visualizing the images and training history.

**Dataset**  
The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits (0-9), each 28x28 pixels in grayscale.

## Methodology

1. **Data Preprocessing:**
   - The MNIST dataset is loaded and normalized, ensuring that each pixel has a value between 0 and 1.
   
2. **Model Architecture:**
   - The model consists of three layers:
     - **Flatten Layer:** Converts the 28x28 pixel images into a 1D array of 784 values (one per pixel).
     - **Dense Hidden Layers:** Two layers with 128 units and ReLU activation to capture the non-linear patterns.
     - **Dense Output Layer:** A final layer with 10 units corresponding to the 10 possible digits (0-9), using a softmax activation function.

3. **Model Training:**
   - The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss function. 
   - The model is trained on the training set for 3 epochs.

4. **Model Evaluation:**
   - The model is evaluated on the test set, and the loss and accuracy are printed.

5. **Model Deployment:**
   - After training, the model is saved to a file. If an existing model is available, it can be loaded for inference.

**Model Inference:**
The trained model is used to predict the digit from custom images, and the predictions are displayed using `Matplotlib`.

## Dataset
- Source: MNIST Dataset (https://www.tensorflow.org/datasets/community_catalog/huggingface/mnist)
- The dataset is automatically loaded and split into training and test sets.

## Model Training and Evaluation

- **Model Architecture:** The model consists of three layers: a flatten input layer, two dense hidden layers, and one dense output layer.
- **Loss Function:** Sparse categorical cross-entropy.
- **Optimizer:** Adam optimizer for efficient training.
- **Metrics:** Accuracy is used to evaluate model performance.
  
The model is trained for 3 epochs and evaluated using the test data. The final loss and accuracy are displayed after evaluation.

## Key Results
- **Model Accuracy:** The model achieves high accuracy (98.6%) in recognizing handwritten digits from the MNIST dataset.
