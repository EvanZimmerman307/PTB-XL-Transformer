# PTB-XL Transformer

An ECG classification project using transformer-based deep learning models for multi-label diagnosis on the [PTB-XL dataset](https://physionet.org/content/ptb-xl/1.0.3/).

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Experiments](#experiments)
- [Results](#results)
- [Project Structure](#project-structure)
- [License](#license)

## 🌟 Overview

This project implements hybrid CNN/ResNet + transformer architectures for ECG signal classification, specifically targeting the superdiagnostic multi-label classification task on the PTB-XL dataset. The project combines convolutional neural networks (CNNs) and transformer encoders to process 12-lead ECG signals and predict diagnostic categories.

The project includes extensive experimentation with different model configurations, data augmentations, and training strategies to optimize classification performance on ECG diagnostics.

## ✨ Features

- **Multi-label ECG Classification**: Supports prediction of multiple diagnostic categories simultaneously
- **Transformer-based Models**: Leverages self-attention mechanisms for ECG sequence processing
- **ResNet/CNN Integration**: Combines ResNet blocks and CNN blocks with transformer encoders
- **Data Augmentation**: Implements noise injection, amplitude scaling, and baseline wander simulation
- **Comprehensive Evaluation**: Uses macro AUC and per-class AUC metrics
- **Cross-validation**: 10-fold stratified cross-validation on PTB-XL dataset
- **Pretrained Models**: Includes trained models achieving high performance

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.5+
- CUDA-capable GPU (recommended for training)

### Dependencies

Install the required packages:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn tqdm wfdb
```

### Dataset Setup

1. Download the PTB-XL dataset from PhysioNet at this link (https://physionet.org/content/ptb-xl/1.0.3/):

2. Place the extracted `ptb-xl/` folder in the project root directory.

## 📊 Dataset

The project uses the [PTB-XL dataset](https://physionet.org/content/ptb-xl/), a large publicly available ECG dataset containing:

- 21,837 ECG recordings from 18,885 patients
- 12-lead ECG signals sampled at 100 and 500 Hz
- 10-second recordings
- Multi-label annotations for cardiac conditions
- 5 superdiagnostic classes:
  - NORM (Normal ECG)
  - MI (Myocardial Infarction)
  - STTC (ST/T Changes)
  - CD (Conduction Disturbance)
  - HYP (Hypertrophy)

## 🚀 Usage

### Running Experiments

Run the main experiment (training and eval with data augmentation):

```bash
python main.py
```

### Replicating Specific Experiments

Experiments can be replicated and run individually using the experiment class:

```python
# In main.py
from experiment import Experiment

# Example: Run training and eval with hybrid CNN/Transformer architecture
experiment = Experiment('<experiment name>', 'superdiagnostic', 'ptb-xl/')
experiment.prepare()
experiment.train()
experiment.predict()
experiment.evaluate()
```

```Python
# In experiment.py modify train() to choose your architecture

def train(self):
    ...
    self.model = ECGHybridNet(input_dim=self.input_shape[1], num_classes=n_classes)
```

### Model Evaluation

**Evaluate predictions** on pretrained models:

```bash
python eval.py
```

### Custom Prediction

Use pre-trained model to generate **predictions and evaluation**:

```bash
python predict.py
```

## 🏗️ Model Architectures

### ResNet-Transformer (Current Best Model)
- **Convolutional Backbone**: ResNet blocks with 1D convolutions
- **Transformer Encoder**: Stack of transformer layers with multi-head attention
- **Classification Head**: Adaptive average pooling + fully connected layers
- **Input**: 12-lead ECG (12 × 1000 samples, 10 second samples at 100 Hz)
- **Output**: 5-class multi-label probabilities

### CNN-Transformer Hybrid
- **Convolutional Backbone**: Simple CNN layers for feature extraction
- **Transformer Encoder**: Self-attention based seq2seq processing
- **Positional Encoding (Optional)**: Sinusoidal positional embeddings
- **Attention Pooling**: Learnable attention weights (experimented)

### Key Components
- **1D ResNet Blocks**: Residual connections with batch normalization
- **Multi-Head Attention**: 8 heads, feed-forward dimension 4× model dimension
- **Dropout**: 0.5 in classification layers
- **Class Balanced Loss**: BCEWithLogitsLoss with positive weights

## 🔬 Experiments

The project includes 14 different experiments optimizing model architecture and training:

### Architecture Experiments (Exp 1-9)
- **Exp 1-3**: Initial CNN-Transformer configurations with different loss functions
- **Exp 4**: Larger model (d_model=256, layers=4) with reduced learning rate
- **Exp 5**: Added convolutional layer and max pooling for better sequence representation
- **Exp 6-7**: Model size reduction and reintroduction of positional encoding
- **Exp 8-9**: Attention pooling mechanisms vs. average pooling

### ResNet Integration (Exp 10-14)
- **Exp 10**: Full ResNet backbone, best performing at Exp 4 scale
- **Exp 11**: Smaller ResNet with increased dropout
- **Exp 12**: Larger ResNet (d_model=256, layers=3)
- **Exp 13-14**: Added data augmentation (noise, scaling, baseline wander)

Saved model weights are stored in `saved_models/`
## 📈 Results

### Best Performance Achievements

| Experiment | Train Macro AUC | Test Macro AUC |
|------------|-----------------|----------------|
| Exp 6 (Best CNN-Transformer) | 0.970 | 0.913 |
| Exp 10 (First ResNet) | 0.982 | 0.901 |
| Exp 12 (Medium ResNet - Best Model) | 0.965 | 0.917 |
| Exp 14 (Final Model) | 0.957 | 0.914 |

### Performance Metrics
- **Macro AUC**: Measures overall multi-label classification performance
- **Class AUC**: Per-class AUC scores for NORM, MI, STTC, CD, HYP
- **Cross-validation**: 10-fold stratified evaluation on PTB-XL

### Data Augmentation Impact
- Noise injection (σ=0.02): Improved robustness
- Amplitude scaling (0.9-1.1): Simulates body type variations
- Baseline wander: Models breathing/cardiac motion artifacts

## 📁 Project Structure

```
├── main.py                    # Main entry point for experiments
├── experiment.py              # Experiment class and data 
├── utils.py                   # Data loading, preprocessing, 
├── eval.py                    # Model evaluation utilities
├── predict.py                 # Prediction on new data
├── vit_main.py               # ViT model experiments
├── vit_experiment.py          # ViT experiment class
├── vit_ecg_dataset.py         # ViT dataset handling
├── models/                    # Model architectures
│   ├── resnet_transformer.py  # ResNet-Transformer model
│   ├── ecg_hybrid_net.py      # Hybrid CNN-Transformer
│   ├── vit.py                 # Vision Transformer
│   └── vit_test.py           # ViT test implementations
├── saved_models/              # Trained model checkpoints
├── ptb-xl/                   # PTB-XL dataset
├── copy_best_models.py       # Model management script
├── count.py                  # Dataset statistics
└── notes.txt                 # Experiment log and notes
```

## 📄 License

This project uses the PTB-XL dataset which is available under PhysioNet's open licensing terms. The code is provided for research and educational purposes.

Please cite the dataset appropriately:
- PTB-XL: Wagner, P. et al. "PTB-XL, a large publicly available electrocardiography dataset." Scientific Data 7.1 (2020): 154.

---

**Note**: This project is intended for research purposes and should not be used for clinical diagnosis without proper validation and regulatory approval.
