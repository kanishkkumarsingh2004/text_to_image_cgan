# 🎨 Text-to-Image Generation with Memory-Optimized CGAN

## Abstract
This project implements a memory-optimized Conditional Generative Adversarial Network (CGAN) for text-to-image generation. The system is designed to generate high-quality 128x128 resolution images based on text descriptions while maintaining efficient GPU memory usage. The implementation supports popular image datasets including CIFAR10 and CIFAR100, with several key optimizations:

- Memory-efficient architecture with reduced network dimensions
- CUDA memory management techniques including expandable segments
- Batch size optimization for better GPU utilization
- Image resolution scaling to balance quality and performance
- Spectral normalization for stable training

The project includes three main components: data processing, model training, and image generation, each implemented as separate Python modules. The system is designed to run on consumer-grade GPUs while maintaining good performance and image quality.
## 🌟 Overview

This repository implements a state-of-the-art **Conditional Generative Adversarial Network (CGAN)** optimized for **text-to-image generation** with enhanced memory efficiency. The model generates high-quality 128x128 images conditioned on text descriptions, supporting popular datasets like CIFAR10 and CIFAR100.

Key Features:
- 🚀 **Memory-Efficient Architecture**: Optimized for GPUs with reduced memory footprint
- 🎯 **Text Conditioning**: Advanced text embedding system for precise image generation
- ⚡ **Training Optimization**: Customizable parameters with GPU acceleration
- 🖼️ **Interactive Interface**: Command-line based image generation with real-time feedback

## 🧠 Architecture

The system is built on three core modules:

### 1. Data Processing Module (`data.py`)
- 🗂️ Dataset loading and preprocessing pipelines
- 🧮 Memory-efficient Generator and Discriminator architectures
- ⚙️ CUDA optimization and training parameters
- 🔤 Text embedding and image transformation pipelines

### 2. Training Module (`traaining.py`)
- 🔄 Adversarial training loop implementation
- 💾 GPU memory management strategies
- 📦 Model checkpointing and spectral normalization removal
- 📉 Loss calculation and backpropagation mechanisms

### 3. Inference Module (`test.py`)
- 🖌️ Interactive image generation interface
- 🔍 Text-to-image translation
- 🎨 Image denormalization and visualization
- 📦 Batch generation with user-defined text prompts

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.6.0 with CUDA 11.8 support
- torchvision 0.21.0
- matplotlib 3.10.0
- numpy 2.1.2

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/kanishkkumarsingh2004/text_to_image_cgan
   cd text-to-image-cgan
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv cgan-env
   source cgan-env/bin/activate  # On Windows use: cgan-env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Training the Model
1. To start training the model, run the following command in your terminal:
   ```bash
   python3 traning.py
   ```
2. To start Generating the Image, run the following command in your terminal:
   ```bash
   python3 test.py
   ```
