# Financial-Distress-Prediction-and-Risk-Modeling
Corporate Bankruptcy Prediction leveraging DNN + Gaussian Naive Bayes

## Overview

This project aims to forecast the probability of a company going bankrupt using its financial indicators. The framework combines a **Deep Neural Network (DNN)** and a **Gaussian Naive Bayes (GNB)** classifier, leveraging both deep feature extraction and probabilistic reasoning. By integrating both models via an ensemble strategy, it achieves robust performance on a highly imbalanced dataset.

---

##  Features

### Exploratory Data Analysis (EDA)

- Analyzed data patterns, feature correlations, and highlighted severe class imbalance.

## Data Preprocessing

- **Feature Selection**: Selected top features using **ANOVA F-scores** to improve relevance and reduce noise.
- **Handling Class Imbalance**: Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to oversample the minority (bankrupt) class.
- **Standardization**: Scaled features to zero mean and unit variance using `StandardScaler`.

---

## Model Architecture

### Deep Neural Network (DNN)

- **Input Layer**: Takes selected financial features as input.
- **Hidden Layers**:
  - Layer 1: 256 neurons, ReLU activation, Batch Normalization, Dropout (50%).
  - Layer 2: 128 neurons, ReLU activation, Batch Normalization, Dropout (50%).
  - Layer 3: 64 neurons, ReLU activation, Batch Normalization, Dropout (40%).
- **Output Layer**: Single neuron with sigmoid activation for binary prediction.
- **Optimizer**: Adam (learning rate = 0.0005).
- **Loss Function**: Binary cross-entropy.

### Gaussian Naive Bayes (GNB)

- Probabilistic model assuming features are conditionally independent and normally distributed.
- Computes posterior probabilities for bankruptcy risk assessment.

---

##  Ensemble Approach

- Combined predictions from **DNN and GNB** using **soft voting**.
- Fine-tuned the final decision threshold to maximize F1-score.

---

##  Dataset

- **Source**: `Train.csv`
- **Size**: 5,455 rows × 96 columns.
- **Target Variable**: `Bankrupt?` (0 = Non-bankrupt, 1 = Bankrupt).
- **Class Distribution**:
  - Non-bankrupt: 5,301 (97.2%)
  - Bankrupt: 154 (2.8%)

---

##  Data Preprocessing Steps

- **Feature Selection**: Kept the top 30 features identified using ANOVA F-score analysis.
- **SMOTE**: Addressed class imbalance by oversampling the minority class.
- **Standardization**: Used `StandardScaler` to ensure features are on the same scale.

---

## Evaluation Metrics

The model was tested on a 20% hold-out split, using a dataset with a heavy imbalance ratio (approx. 1:33). Performance metrics:

- **Accuracy**: 97%
- **Precision**: 46.34%
- **Recall**: 61.29%
- **F1-Score**: 52.78%
- **Best Threshold**: 0.41

---

## Conclusion

This hybrid ensemble strategy effectively leverages both deep neural feature learning and probabilistic classification to address the challenging problem of bankruptcy prediction. Despite severe class imbalance, the combined approach achieved a balanced trade-off between precision and recall.
