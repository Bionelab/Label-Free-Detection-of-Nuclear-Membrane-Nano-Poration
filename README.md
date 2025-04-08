
# Cell Morphology Analysis Framework

This repository contains code for analyzing cellular morphology using orientation-invariant Variational Autoencoders, a deep learning approach for unsupervised feature extraction from microscopy images.

## Setup Instructions

### Prerequisites
- Python 3.9+
- PyTorch 1.7+
- CUDA-compatible GPU (recommended)
- Weights & Biases account for experiment tracking

### Installation

```bash
git clone https://github.com/username/repository.git
cd repository
pip install -r requirements.txt
```

## Workflow Overview

The pipeline consists of three main stages:
1. **Data Acquisition**: Download necessary datasets and pre-trained models
2. **Data Preparation**: Process raw microscopy images for model input
3. **Model Training & Analysis**: Train the model and analyze cell morphology features

## 1. Data Acquisition

First, run the following download scripts in order:

```bash
# Download raw image data and polygon annotations
python PREP/download_images_polys.py

# Download the MEFs dataset (public dataset for training)
python VAE/data/download_MEFs.py

# Download pre-trained model checkpoints
python VAE/model_training_checkpoints/download_saved_model.py
```

## 2. Data Preparation

This stage prepares the raw microscopy images for input to the model:

```bash
# Process raw images and create model-ready datasets
python PREP/prepare_images.py
```

This script segments cells, normalizes images, and organizes the data for the VAE input pipeline.

## 3. Model Training & Analysis

### Training the Model

Train the orientation-invariant VAE model using the MEFs dataset:

```bash
# Train the model on MEFs dataset
python VAE/run.py --config configs/config_MEFs.py
```

### Applying to Your Data

Apply the trained model to your prepared image dataset:

```bash
# Run inference on prepared images
python VAE/inference.py --model_path path/to/checkpoint --data_dir path/to/prepared_images
```

## Analysis Notebooks

Run the following notebooks in order to analyze your results:

### 1. Feature Extraction and Visualization (`analysis/01_feature_extraction.ipynb`)

This notebook handles:
- Loading trained models from checkpoints
- Extracting latent space representations from cell images
- Generating UMAP and t-SNE visualizations of the latent space
- Visualizing reconstructed images to assess model quality
- Creating latent space traversals to understand feature meanings

Run with:
```bash
jupyter notebook analysis/01_feature_extraction.ipynb
```

### 2. Morphology Analysis (`analysis/02_morphology_analysis.ipynb`)

This notebook provides:
- Clustering analysis using K-means and HDBSCAN on latent features
- Statistical analysis of morphological features across conditions
- Correlation of latent features with known biological attributes
- Outlier detection to identify rare or abnormal cell morphologies
- Heatmap visualization of feature distributions across experimental conditions

Run with:
```bash
jupyter notebook analysis/02_morphology_analysis.ipynb
```

### 3. Temporal Analysis (`analysis/03_temporal_analysis.ipynb`)

For time-series experiments:
- Tracks morphological changes over time
- Creates trajectory visualizations in latent space
- Identifies temporal patterns in morphological changes
- Quantifies rate of change for different experimental conditions

Run with:
```bash
jupyter notebook analysis/03_temporal_analysis.ipynb
```

### 4. Condition Comparison (`analysis/04_condition_comparison.ipynb`)

For comparing different experimental conditions:
- Differential analysis between treatment and control groups
- Visualization of condition-specific morphological features
- Statistical testing for significant differences between conditions
- Generation of morphological signatures for each condition

Run with:
```bash
jupyter notebook analysis/04_condition_comparison.ipynb
```

## Experiment Tracking

This codebase uses Weights & Biases (wandb) for experiment tracking and visualization. Key metrics logged include:
- Reconstruction loss
- KL divergence
- Training/validation metrics
- Latent space visualizations

To view your experiments, log in to your wandb account in your browser after running training scripts.

## Project Structure


├── PREP/                      # Data preparation scripts
│   ├── download_images_polys.py
│   └── prepare_images.py
├── VAE/                       # VAE implementation
│   ├── data/                  # Data loading utilities
│   │   └── download_MEFs.py
│   ├── models/                # Model architecture definitions
│   ├── model_training_checkpoints/
│   │   └── download_saved_model.py
│   ├── train_loops.py         # Training loop implementation
│   └── run.py                 # Main training script
├── analysis/                  # Analysis notebooks
│   ├── 01_feature_extraction.ipynb
│   ├── 02_morphology_analysis.ipynb
│   ├── 03_temporal_analysis.ipynb
│   └── 04_condition_comparison.ipynb
└── configs/                   # Configuration files
```

## Acknowledgements

The orientation-invariant VAE implementation in the VAE directory is based on the O2VAE model from [jmhb0/o2vae](https://github.com/jmhb0/o2vae), which provides a framework for orientation-invariant representation learning in cell biology data.

