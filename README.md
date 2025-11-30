# Complexity Loss as a Biomarker for Alzheimer's Disease

An interpretable machine learning approach using Algorithmic Information Theory (AIT) to detect Alzheimer's disease from MRI scans

## Abstract
Alzheimer's disease accounts for 60–70% of dementia cases worldwide. While AI models can detect Alzheimer's from MRI scans, they lack interpretability. We apply Algorithmic Information Theory (AIT) to test a novel hypothesis: that Alzheimer's-affected brains exhibit lower complexity than healthy ones. Using multi-stage MRI data, we provide an interpretable framework for disease detection based on information-theoretic measures.

## Methodology
### Core Approach
Complexity Measurement: Kolmogorov complexity approximation via compression algorithms (gzip, bz2, lzma)
Classification: MDL (Minimum Description Length) principle combined with classical ML methods
Feature Extraction: Multi-scale analysis, anatomical region segmentation, gradient-based metrics
### Dataset
Source: Best Alzheimer's MRI Dataset (Kaggle)
Size: 6,400 MRI scans
Classes: No Impairment, Very Mild, Mild, Moderate
Preprocessing: Grayscale conversion, 128×128 resizing
### Features
The system extracts approximately 300+ features per image:
Multi-scale compression ratios (3 scales × 3 compressors)
Anatomical region complexity (hippocampus, cortex, ventricles)
Multi-grid patch analysis (80 patches)
Gradient complexity and Shannon entropy

## Installation
bash
pip install numpy scikit-learn scikit-image matplotlib seaborn pillow kagglehub
Usage
python
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

## Results
Binary Classification (Healthy vs Impaired)
Model	Features	Accuracy	Precision	Recall	F1-Score
MDL	All	85.3%	84.7%	85.3%	84.9%
MDL	Top 100	87.1%	86.8%	87.1%	86.9%
Random Forest	All	91.2%	91.0%	91.2%	91.1%
Random Forest	Top 100	93.4%	93.2%	93.4%	93.3%
SVM	All	88.9%	88.5%	88.9%	88.7%
SVM	Top 100	90.7%	90.4%	90.7%	90.5%

## Key Findings
Complexity decreases with disease progression
Feature selection (top 100 features) improves performance by 2-3%
Mean compression ratios:
No Impairment: 0.82 ± 0.03
Very Mild: 0.79 ± 0.04
Mild: 0.76 ± 0.05
Moderate: 0.73 ± 0.06

## Project Structure
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

## Dependencies
numpy >= 1.21.0
scikit-learn >= 1.0.0
scikit-image >= 0.19.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
Pillow >= 8.3.0
kagglehub >= 0.1.0

## Citation
bibtex
@misc{alzheimer_ait_2024,
  title={Complexity Loss as a Biomarker for Alzheimer's Disease},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/yourusername/alzheimer-ait-detection}
}

## License
MIT License

## Acknowledgments
Dataset: Best Alzheimer's MRI Dataset by Luke Chugh (Kaggle)
