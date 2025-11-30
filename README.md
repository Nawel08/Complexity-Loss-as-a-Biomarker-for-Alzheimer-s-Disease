# Complexity Loss as a Biomarker for Alzheimer's Disease

An interpretable machine learning approach using Algorithmic Information Theory (AIT) to detect Alzheimer's disease from MRI scans.

## Abstract

Alzheimer's disease accounts for 60–70% of dementia cases worldwide. While AI models can detect Alzheimer's from MRI scans, they lack interpretability. We apply Algorithmic Information Theory (AIT) to test a novel hypothesis: that Alzheimer's-affected brains exhibit lower complexity than healthy ones. Using multi-stage MRI data, we provide an interpretable framework for disease detection based on information-theoretic measures.

## Methodology

**Core Approach**

- Complexity Measurement: Kolmogorov complexity approximation via compression algorithms (gzip, bz2, lzma)
- Classification: MDL (Minimum Description Length) principle combined with classical ML methods
- Feature Extraction: Multi-scale analysis, anatomical region segmentation, gradient-based metrics

**Dataset**

- Source: Best Alzheimer's MRI Dataset (Kaggle)
- Size: 11,519 MRI scans
- Classes: No Impairment (27.8%), Very Mild (26.1%), Mild (23.8%), Moderate (22.3%)
- Binary split: Healthy (53.9%), Impaired (46.1%)
- Preprocessing: Grayscale conversion, 128×128 resizing

**Features**

The system extracts approximately 300+ features per image:
- Multi-scale compression ratios (3 scales × 3 compressors)
- Anatomical region complexity (hippocampus, cortex, ventricles)
- Multi-grid patch analysis (80 patches)
- Gradient complexity and Shannon entropy

## Installation

```bash
pip install numpy scikit-learn scikit-image matplotlib seaborn pillow kagglehub
```

## Usage

```python
# Load dataset
images, labels, label_names = load_mri_dataset_from_kaggle()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

# Extract features
X_train_features = extract_features_dataset(X_train)
X_test_features = extract_features_dataset(X_test)

# Feature selection
selector = SelectKBest(f_classif, k=100)
X_train_sel = selector.fit_transform(X_train_features, y_train)
X_test_sel = selector.transform(X_test_features)

# Train classifier
y_pred = mdl_classifier_features(X_train_sel, y_train, X_test_sel)
```

## Results

### Binary Classification (Healthy vs Impaired)

| Model | Features | Accuracy | Precision | Recall | F1-Score |
|-------|----------|----------|-----------|--------|----------|
| MDL | All | 57.60% | 59.31% | 57.60% | 57.16% |
| MDL | Top 100 | 65.54% | 67.02% | 65.54% | 65.39% |
| Logistic Regression | All | 90.58% | 90.94% | 90.58% | 90.52% |
| Logistic Regression | Top 100 | 89.15% | 89.41% | 89.15% | 89.09% |
| Random Forest | All | 94.05% | 94.63% | 94.05% | 94.01% |
| Random Forest | Top 100 | 93.88% | 94.43% | 93.88% | 93.83% |
| SVM | All | 89.89% | 90.73% | 89.89% | 89.77% |
| SVM | Top 100 | 96.18% | 96.23% | 96.18% | 96.17% |

### Key Findings

- **Best Performance**: SVM with Top 100 features achieves 96.18% accuracy
- **Cross-Validation Results** (5-Fold):
  - Random Forest: 92.77% ± 0.76%
  - Logistic Regression: 89.91% ± 0.47%
  - SVM: 88.74% ± 0.76%
- **Classical ML vs AIT**: Traditional ML models (Random Forest, SVM) significantly outperform the MDL approach
- **Feature Selection Impact**: 
  - MDL benefits most from feature selection (+7.94% accuracy)
  - SVM improves significantly with feature selection (+6.29% accuracy)
  - Random Forest and Logistic Regression remain stable
- **Complexity Analysis**: Hypothesis partially validated
  - Healthy (No + Very Mild): 0.5783 ± 0.0233
  - Impaired (Mild + Moderate): 0.5914 ± 0.0200
  - Note: Higher compression ratio indicates lower complexity (contrary to initial hypothesis)
  - Complexity progression is non-monotonic across 4-class stages
- **Interpretability Trade-off**: While MDL provides theoretical interpretability through AIT, classical ML methods achieve superior predictive performance

## Project Structure

```
alzheimer-ait-detection/
├── notebooks/
│   └── AIT_Alzheimer_Analysis.ipynb
├── src/
│   ├── complexity_measures.py
│   ├── data_loading.py
│   ├── feature_extraction.py
│   ├── classifiers.py
│   └── visualization.py
└── requirements.txt
```

## Dependencies

- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- scikit-image >= 0.19.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- Pillow >= 8.3.0
- kagglehub >= 0.1.0

## Citation

```bibtex
@misc{alzheimer_ait_2024,
  title={Complexity Loss as a Biomarker for Alzheimer's Disease},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/yourusername/alzheimer-ait-detection}
}
```

## License

MIT License

## Acknowledgments

Dataset: Best Alzheimer's MRI Dataset by Luke Chugh (Kaggle)
