# Tuberculosis-classification
# Tuberculosis Detection using Ensemble Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A robust deep learning framework for tuberculosis disease detection using ensemble classification on chest X-ray (CXR) images. The system combines predictions from MobileNetV2, a custom CNN, and Vision Transformer (ViT) models using majority voting for enhanced accuracy.

## 🎯 Overview

This research develops an automated tuberculosis screening system that achieves **99% accuracy** through ensemble learning. The framework classifies chest X-ray images into two categories: **Normal** and **Tuberculosis**, providing a reliable tool for early TB diagnosis and patient care.

## 📊 Performance Results

| Model | Without Augmentation | With Augmentation |
|-------|---------------------|-------------------|
| MobileNetV2 | 98.17% | 97.45% |
| Custom CNN | 95.42% | 94.60% |
| Swin Transformer | 97.25% | 97.45% |
| **Ensemble (Voting)** | **98.98%** | **99.00%** |

## 🗂️ Dataset

The dataset combines multiple publicly available chest X-ray sources, totaling over 24,000 images after augmentation:

### Dataset Composition
| Split | Normal | Tuberculosis |
|-------|--------|-------------|
| Train | 4,907 | 5,971 |
| Validation | 548 | 434 |
| Test | 548 | 434 |
| **Augmented Train** | **11,803** | **12,925** |

### Data Sources
1. [Tuberculosis (TB) Chest X-ray Database](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) - Qatar University & University of Dhaka
2. [Chest X-ray (Normal/Pneumonia/COVID-19/Tuberculosis)](https://www.kaggle.com/datasets/jtiptj/chest-xray-pneumoniacovid19tuberculosis)
3. [Chest X-rays Tuberculosis from India](https://www.kaggle.com/datasets/raddar/chest-xrays-tuberculosis-from-india) - Jaypee University
4. [Combined Unknown Pneumonia and Tuberculosis](https://www.kaggle.com/datasets/rifatulmajumder23/combined-unknown-pneumonia-and-tuberculosis)
5. [Pneumonia, Tuberculosis, and Normal Chest X-rays](https://www.kaggle.com/datasets/rupeshmahanty/pneumonia-tuberculosis-normal)

## 🏗️ Architecture

### Model Components

#### 1. MobileNetV2
- **Input Size**: 224×224×3
- **Architecture**: Pre-trained with depthwise separable convolutions
- **Features**: Global average pooling + Dense layers (128 units)
- **Regularization**: Dropout (0.3, 0.2) + Batch normalization

#### 2. Custom CNN
- **Input Size**: 64×64×3
- **Architecture**: ResNet-inspired with residual connections
- **Layers**: Convolutional blocks (64→128→256→512 filters)
- **Features**: Global average pooling + Dense layers (256→128 units)
- **Regularization**: Dropout (0.4, 0.3) + Batch normalization

#### 3. Swin Transformer
- **Input Size**: 224×224×3
- **Architecture**: Swin-Base with shifted window attention
- **Pre-training**: ImageNet pre-trained
- **Fine-tuning**: Classifier + last encoder layer unfrozen

### Ensemble Method
- **Strategy**: Majority voting
- **Combination**: Aggregates predictions from all three models
- **Output**: Final binary classification (Normal/Tuberculosis)

## 🚀 Quick Start

### Prerequisites
