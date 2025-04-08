# Cell Morphology Analysis for Nuclear Membrane Nano-Poration Detection

This repository contains code for "Label-Free Detection of Nuclear Membrane Nano-Poration" research, using orientation-invariant Variational Autoencoders for unsupervised feature extraction from microscopy images to detect nuclear membrane disruptions.

## Repository Structure

The analysis pipeline is organized into two main Jupyter notebooks and supporting Python modules:

```
├── 1_run_to_convert_images.ipynb  # Image preprocessing and Ku-80 analysis
├── 2_main_notebook.ipynb          # VAE model application and SVM classification
├── VAE/                           # Variational Autoencoder implementation
│   ├── data/                      # Data utilities
│   ├── feature_change/            # Feature manipulation tools
│   ├── models/                    # Model architecture
│   ├── model_training_checkpoints/# Pretrained model weights
│   ├── run.py                     # Model training utilities
│   └── train_loops.py             # Training loop implementation
├── ML/                            # Machine learning utilities
│   ├── train_svm.py               # SVM training and evaluation
│   └── svm_config.py              # SVM configuration parameters
└── PREP/                          # Data preparation utilities
    ├── analyzer.py                # Image analysis functions
    ├── line_method.py             # Ku-80 line profile analysis
    ├── plot_utils.py              # Visualization utilities
    ├── download_images_polys.py   # Data download functions
    └── utils.py                   # General utilities
```

## Analysis Workflow

The analysis pipeline is executed through the two Jupyter notebooks:
### 0. Using biodock [https://login.biodock.ai/] 

- to segment cells and nuclei with the labels 'cell' and 'nuc',
- and download the corresponding polygons.
- here we have our study polygons availble
  
### 1. Image Preprocessing and Ku-80 Analysis (`1_run_to_convert_images.ipynb`)

This notebook handles:
- Loading microscopy images of cells on nanopillars
- finding the pairs of nucleus and cell
- Extracting Ku-80 and DAPI intensity profiles
- Calculating MSE between Ku-80 and DAPI to identify nuclear membrane poration
- Converting segmented polygons to binary masks
- Exporting processed data for model training

### 2. VAE Model and SVM Classification (`2_main_notebook.ipynb`)

This notebook performs:
- Training orientation-invariant VAE on MEFs dataset or Loading the pretrained orientation-invariant VAE model
- Extracting 32-dimensional embeddings for cells and nuclei on the prepared images in step1
- Combining relative morphological features [relative size and locations of cell and nucleus pairs] with VAE embeddings
- Training an SVM classifier to predict nuclear poration
- Evaluating model performance with ROC and precision-recall curves
- Feature importance analysis using SHAP
- Visualizing morphological features that correlate with nuclear poration

## How to Run

1. Clone the repository and install dependencies:
```bash
git clone https://github.com/Bionelab/Label-Free-Detection-of-Nuclear-Membrane-Nano-Poration.git
cd Label-Free-Detection-of-Nuclear-Membrane-Nano-Poration
pip install -r requirements.txt
```

2. Run the first notebook to process images:
```bash
jupyter notebook 1_run_to_convert_images.ipynb
```

3. Run the second notebook for model application and classification:
```bash
jupyter notebook 2_main_notebook.ipynb
```

## Key Technologies

- **Image Processing**: Polygon-based segmentation of cell and nucleus boundaries
- **Feature Extraction**: Orientation-invariant VAE based on the implementation from [jmhb0/o2vae](https://github.com/jmhb0/o2vae)
- **Classification**: SVM with optimized hyperparameters
- **Interpretation**: SHAP analysis for feature importance
- **Visualization**: Custom plotting utilities for morphological analysis

## Citation
the orientation-invariant VAE code, is cloned from [https://github.com/jmhb0/o2vae]. We thank them, for the clear instuction and code

If you use this code in your research, please cite our paper:
- **Code**:
- [![DOI](https://sandbox.zenodo.org/badge/941298210.svg)](https://handle.stage.datacite.org/10.5072/zenodo.196087)

- **Paper**:
@article{rahmani2024labelfree,
  title={Label-Free Detection of Nuclear Membrane Nano-Poration},
  author={Rahmani, Keivan and Sadr, Leah and Sarikhani, Einollah and Naghsh-Nilchi, Hamed and Onwuasoanya, Chichi and Wong, Yu Ching and Wen, Weijia and Jahed, Zeinab},



  journal={},
  year={2024}
}
```
