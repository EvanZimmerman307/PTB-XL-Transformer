# ECG Classification with Transformer Models: Progress Report

## Motivation

Accurate ECG interpretation is critical for cardiac diagnosis, yet the lack of large, well-labeled ECG datasets poses significant challenges for ML/AI research. Traditional CNN-based approaches process ECG signals as 1D time series, but may not optimally capture complex temporal dependencies across long sequences. Transformer architectures, with their self-attention mechanisms, offer the potential to better model relationships between distant ECG segments, potentially improving diagnostic accuracy for multi-label cardiac conditions.

## The PTB-XL Dataset

My work utilizes the PTB-XL dataset, a large publicly available electrocardiography collection containing 21,837 12-lead ECG recordings from 18,885 patients. Each recording is a 10-second signal sampled at 100 Hz, annotated with multi-label diagnostic classes aggregated into 5 superdiagnostic categories: NORM (normal ECG), MI (myocardial infarction), STTC (ST/T changes), CD (conduction disturbance), and HYP (hypertrophy). This comprehensive dataset enables robust evaluation through 10-fold stratified cross-validation. The 10-folds are created by the dataset creators and widely used throughout research on this dataset.

## Methodology and Results

I developed a series of transformer-based models for multi-label ECG classification, conducting 14 systematic experiments to optimize architecture and training strategies. Starting with hybrid CNN-Transformer models, I progressively improved performance through architectural modifications: increased model capacity, integration of ResNet blocks for better feature learning, and strategic dropout placement.

The best ResNet-Transformer architecture achieved strong performance with 91.7% test macro AUC, demonstrating competitive results for the superdiagnostic classification task. Key contributions included implementing class-balanced loss functions to handle label imbalance and developing effective data augmentation strategies (noise injection, amplitude scaling, and baseline wander simulation) that improved model robustness.

## Next Steps

### Finer-Grained Diagnostic Classification

While the current work focused on superdiagnostic categories, the PTB-XL dataset's detailed annotation hierarchy enables exploration of more specific cardiac conditions. Future work will investigate classifying subtypes of myocardial infarction (e.g., anterior vs. inferior MI) and ST/T changes (e.g., ischemic vs. strain patterns), potentially providing more actionable diagnostic information for clinical decision-making.

### Vision Transformer Fine-Tuning

The promising results from the hybrid CNN-Transformer training motivated exploration of pre-trained Vision Transformers (ViT). In my last steps, I began adapting ViT models by converting ECG signals into spectrogram representations and stacking 12-lead spectrograms into multi-channel images suitable for ViT processing. This approach could leverage any large open source ViT models pretrained weights, with modification of the patch embedding layer to accommodate the ECG-derived spectrograms. This transfer learning strategy shows promise for ECG classification tasks.


## Useful References

1. **Paper with baseline deep learning methods and results**:

- Strodthoff, N., Wagner, P., Schaeffter, T., Samek, W. (2021). "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL". IEEE Journal of Biomedical and Health Informatics 25, no. 5, 1519-1528.
- https://arxiv.org/abs/2004.13701
- https://github.com/helme/ecg_ptbxl_benchmarking/tree/master

2. **Recent survey on Transformers for ECG Diagnosis**:
- Ansari, M.Y., Yaqoob, M., Ishaq, M. et al. A survey of transformers and large language models for ECG diagnosis: advances, challenges, and future directions. Artif Intell Rev 58, 261 (2025). https://doi.org/10.1007/s10462-025-11259-x
