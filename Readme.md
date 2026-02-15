# Learning Probability Density Function using GAN

## Overview

This project aims to learn the Probability Density Function (PDF) of a transformed air quality variable using a Generative Adversarial Network (GAN). The GAN learns the distribution directly from data samples without assuming any predefined mathematical form of the distribution.

The dataset used contains air quality measurements from various locations in India. The NO₂ concentration feature is selected and transformed to create a new random variable whose PDF is unknown. The GAN is then trained to estimate this PDF.

---

## Objective

The main objectives of this project are:

• To transform the NO₂ concentration using a nonlinear transformation
• To train a GAN to learn the probability density function of the transformed variable
• To generate synthetic samples from the learned distribution
• To compare the real and generated probability density functions

---

## Dataset

Dataset Name: India Air Quality Dataset
Source: Kaggle

Feature used:

NO₂ concentration

Dataset link:
https://www.kaggle.com/datasets/shrutibhargava94/india-air-quality-data

---

## Data Transformation

The original variable x represents the NO₂ concentration.

A nonlinear transformation is applied to obtain a new variable z:

z = x + a_r sin(b_r x)

where:

a_r = 0.5 × (r mod 7)
b_r = 0.3 × ((r mod 5) + 1)

r = 102317059

This transformation produces a new random variable whose probability density function must be learned.

---

## Transformation Parameters

Roll Number: YOUR_ROLL_NUMBER

Example:

a_r = 2.5
b_r = 1.5

These values are calculated using the roll number.

---

## GAN Architecture

The GAN consists of two neural networks:

---

### Generator Network

Purpose: Generate synthetic samples similar to real data

Architecture:

Input Layer: 1 neuron (random noise)
Hidden Layer 1: 16 neurons, ReLU activation
Hidden Layer 2: 16 neurons, ReLU activation
Output Layer: 1 neuron

The generator learns to produce samples that resemble the real data distribution.

---

### Discriminator Network

Purpose: Distinguish between real and generated samples

Architecture:

Input Layer: 1 neuron
Hidden Layer 1: 16 neurons, ReLU activation
Hidden Layer 2: 16 neurons, ReLU activation
Output Layer: 1 neuron, Sigmoid activation

The discriminator learns to classify whether a sample is real or fake.

---

## Training Methodology

The GAN is trained using adversarial training:

Step 1: The generator produces fake samples from random noise

Step 2: The discriminator receives both real and fake samples

Step 3: The discriminator learns to distinguish real from fake samples

Step 4: The generator learns to fool the discriminator

Step 5: This process repeats for multiple epochs

Training details:

Optimizer: Adam
Loss Function: Binary Cross Entropy Loss
Epochs: 2000
Batch Size: 64

---

## Probability Density Function Estimation

After training, the generator produces synthetic samples.

Kernel Density Estimation (KDE) is used to estimate the probability density function from:

• Real samples
• Generated samples

This allows visual comparison between real and learned distributions.

---

## Results

### Result Graph 1: Real Distribution

Shows the probability density function of transformed variable z.

---

### Result Graph 2: GAN Training Loss

Shows generator and discriminator loss over training epochs.

Observation:

• Loss stabilizes over time
• Training is stable

---

### Result Graph 3: Real vs Generated PDF

Shows comparison between:

• Real PDF
• GAN-generated PDF

Observation:

The GAN successfully learns the shape of the distribution.

---

## Result Table

| Metric                  | Observation |
| ----------------------- | ----------- |
| Training Stability      | Stable      |
| Mode Coverage           | Good        |
| Distribution Similarity | High        |
| Convergence             | Achieved    |

---

## Observations

• The generator learns the distribution effectively
• The discriminator improves classification during training
• Generated samples follow the same pattern as real data
• GAN successfully estimates the unknown probability density function

---

## Conclusion

This project demonstrates that Generative Adversarial Networks can effectively learn and estimate the probability density function of a transformed variable directly from data.

The generated distribution closely matches the real distribution, showing that GAN is a powerful tool for density estimation.

---

## Repository Structure

GAN-PDF-Estimation/

│── GAN_PDF_Estimation_NO2.ipynb
│── README.md

---

## Technologies Used

Python
PyTorch
NumPy
Pandas
Matplotlib
Seaborn
Google Colab

---

## Author

Name: Anshika Ahuja
Roll Number: 102317059
