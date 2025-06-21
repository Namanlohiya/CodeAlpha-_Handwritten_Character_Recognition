🧠 Handwritten Character Recognition using CNN (MNIST)

This project is a deep learning-based character recognition system that uses a Convolutional Neural Network (CNN) to classify handwritten digits from the popular **MNIST** dataset. It is built using **TensorFlow** and **Keras**, and achieves high accuracy in recognizing digits (0–9) from 28x28 grayscale images.

## 📌 Project Overview

Handwritten digit recognition is one of the fundamental problems in computer vision. The MNIST dataset is a benchmark dataset consisting of 70,000 grayscale images of handwritten digits (60,000 for training and 10,000 for testing). This project implements a CNN to train on this dataset and evaluate its performance on unseen data.

## 🛠️ Technologies Used

* Python 🐍
* TensorFlow / Keras 📦
* NumPy 📊
* Matplotlib 📈
* MNIST Dataset 📚

## 🧮 Model Architecture

The Convolutional Neural Network (CNN) model architecture:
Input Layer: 28x28x1 grayscale images

1. Conv2D (32 filters, 3x3 kernel, ReLU activation)
2. MaxPooling2D (2x2)
3. Conv2D (64 filters, 3x3 kernel, ReLU activation)
4. MaxPooling2D (2x2)
5. Flatten
6. Dense (64 neurons, ReLU activation)
7. Dense (10 neurons, Softmax activation - for 10 digit classes)

## 📊 Results

* The model achieves an accuracy of over **98%** on the test dataset.
* A sample prediction is visualized using `matplotlib`, displaying both the true and predicted label.
* The trained model is saved as `mnist_cnn_model.h5`
