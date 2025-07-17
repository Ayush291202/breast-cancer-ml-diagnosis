# Breast Cancer Diagnosis using Machine Learning

This project explores multiple machine learning models to predict breast cancer diagnosis (Benign vs. Malignant) using the Wisconsin Breast Cancer Diagnostic dataset. We implement feature engineering, model evaluation, interpretability using LIME, and advanced hyperparameter tuning.

## Dataset

- **Source**: UCI Machine Learning Repository  
- **Name**: Breast Cancer Wisconsin (Diagnostic)  
- **Features**: 30 numeric features extracted from cell nuclei  
- **Target**: Diagnosis (`M` = Malignant, `B` = Benign)

## Models Used

| Model                  | Accuracy | Precision | Recall  | F1 Score | AUC-ROC |
|------------------------|----------|-----------|---------|----------|---------|
| Logistic Regression    | 94.74%   | 0.9737    | 0.8810  | 0.9250   | 0.9921  |
| Random Forest          | 97.37%   | 1.0000    | 0.9286  | 0.9630   | 0.9929  |
| Support Vector Machine | 90.35%   | 1.0000    | 0.7381  | 0.8493   | 0.9808  |
| K-Nearest Neighbors    | 91.23%   | 0.9706    | 0.7857  | 0.8684   | 0.9547  |
| Gradient Boosting      | 96.49%   | 1.0000    | 0.9048  | 0.9500   | 0.9947  |
| Naive Bayes            | 93.86%   | 1.0000    | 0.8333  | 0.9091   | 0.9934  |

## Key Features

- **Data Preprocessing**  
  Missing value check, label encoding, feature scaling

- **Model Building**  
  Logistic Regression, Random Forest, SVM, KNN, Naive Bayes, Gradient Boosting

- **Model Interpretation (Explainability)**  
  Used LIME (Local Interpretable Model-Agnostic Explanations) for individual predictions

- **Model Evaluation**  
  Accuracy, Precision, Recall, F1 Score, ROC-AUC, comparison bar plots

- **Hyperparameter Tuning**  
  Used GridSearchCV for Random Forest and Gradient Boosting

- **(Optional) Fairness Check**  
  You can extend to check model bias against demographic groups (if features are available)

## Visualization

- LIME explanations of local predictions  
- Bar charts comparing evaluation metrics across models  
- Confusion matrices

## Installation

```bash
pip install -r requirements.txt
