# Handwritten Digit Recognition using Neural Network and Pygame Canvas

This project demonstrates a neural network model trained to recognize handwritten digits using the MNIST dataset. The model uses backpropagation to learn and achieve accurate digit recognition. Additionally, this program provides an interactive canvas, built using Pygame, where users can draw a digit, and the model will attempt to recognize the drawn number in real time.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Overview
The goal of this project is to build a simple, interactive application to demonstrate the power of neural networks in recognizing handwritten digits. The project consists of:
1. A neural network model that has been trained on the MNIST dataset using backpropagation to learn how to classify digits (0-9).
2. A Pygame-based canvas where you can draw a digit and see if the model can recognize it.

## Project Structure
The main files in this repository include:

- `train.py`: Contains the code for building and training the neural network model. This model uses backpropagation to adjust its weights based on errors and improve accuracy.
- `main.py`: The main application file. This file includes code to set up the Pygame canvas, take user input (drawn digits), and process that input through the trained model to classify the drawn digit.
- `test.py`: This gives you a test you can do with the AI to see how well it does on the testing examples of the MNIST dataset

## Installation
To run this project locally, you'll need to have Python installed, as well as the following packages:

- `pygame` (for creating the drawing canvas)
- `numpy` (for numerical computations)
- `torchvision` (for accessing the MNIST dataset)
- `pickle` (for saving and running the model)
- `scipy` (for blurring effect on canvas)
  
You can install these dependencies using pip:

```bash
pip install pygame numpy torchvision pickle scipy
```

Or you can install it with conda, which is what I did and recommend:
```bash
conda install -c conda-forge -c pytroch numpy pygame torchvision scipy
```
## Usage
1. Clone the repository:
```bash
git clone https://github.com/HendinTom/PythonNumberAI.git
cd pythonnumberai
```
(Optional but recommended: Create a conda environment)
```bash
conda create --name pythonnumberai python=3.12
```
2. Run main.py to start the application:
```bash
python main.py
```
3. A Pygame window will open, allowing you to draw a digit with your mouse. Once you finish drawing, the model will try to classify your input and display the recognized number on the screen.

## How it Works
Model Training: The neural network is trained on the MNIST dataset, which consists of 60,000 training images of handwritten digits and 10,000 test images. During training, backpropagation is used to minimize the error rate, enabling the model to learn from its mistakes and improve its performance over time.

Drawing and Recognition: The main program creates a Pygame window that acts as a canvas for drawing. After the user draws a digit, the drawing is processed and resized to match the MNIST image format. The image data is then fed to the model, which outputs its prediction of the drawn digit.

## Contributing
Contributions are welcome! If you’d like to improve the model’s accuracy, optimize the drawing interface, or add new features, please fork the repository and submit a pull request.

## Liscence
This project is licensed under the Apache 2.0 License - see the LICENSE file for details.
