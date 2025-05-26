# Predictive Modeling and Risk Stratification in Breast Cancer Diagnosis Using Machine Learning

## Project Objective

To develop, evaluate, and interpret multiple machine learning models to distinguish between benign and malignant breast tumors. The project also identifies key diagnostic biomarkers and applies statistical validation techniques to ensure clinical interpretability — aligning with core principles in **biostatistics** and **clinical data science**.

---

## Dataset

- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset  
- **Features:** 30 numeric features (mean, worst, and standard error of various cell nucleus characteristics)  
- **Target:** `Diagnosis` — `M` (Malignant), `B` (Benign)

---

## Models Implemented

| Model                | Description                               |
|---------------------|-------------------------------------------|
| Logistic Regression | Interpretable baseline model              |
| Random Forest       | Ensemble model with feature importance    |
| Support Vector Machine | Effective in high-dimensional spaces    |
| K-Nearest Neighbors | Instance-based learning                   |
| Gradient Boosting   | Boosted trees for accuracy and robustness |
| Naive Bayes         | Probabilistic model for baseline          |

---

## Evaluation Metrics

Each model was evaluated using **5-Fold Cross-Validation** with the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
- Standard Deviation of all metrics (for statistical robustness)

Bar plots visualize each metric across models, including **error bars** for standard deviation.

---

## Model Interpretability

SHAP (SHapley Additive exPlanations) was used to interpret the **top predictive features**:
- Worst Concave Points
- Mean Radius
- Worst Area
- Worst Perimeter

These features are known clinical biomarkers and help in building **trustworthy** ML systems for healthcare.

---

| Model                | Accuracy (%) | Precision (%) | Recall (%) | F1 Score (%) | AUC (%) |
|----------------------|--------------|---------------|------------|--------------|---------|
| Logistic Regression  | 96.5%        | 97.5%         | 92.9%      | 95.1%        | 95.7%   |
| Gradient Boosting    | 96.5%        | 100.0%        | 90.5%      | 95.0%        | 95.2%   |
| Random Forest        | 95.6%        | 100.0%        | 88.1%      | 93.7%        | 94.0%   |
| KNN                  | 75.4%        | 76.9%         | 47.6%      | 58.8%        | 69.6%   |
| SVM                  | 63.2%      | 0.0%          | 0.0%       | 0.0%         | 50.0%   |
| Naive Bayes          | 62.3%        | 0.0%          | 0.0%       | 0.0%         | 49.3%   |

---

## Clinical Implications

- Designed to assist radiologists/pathologists in early tumor classification.
- SHAP-based interpretation ensures transparency and aligns with **clinical guidelines**.
- Avoids overfitting by incorporating **statistical validation** with standard deviation analysis.

---

## Future Work

- Hyperparameter tuning using GridSearchCV or Optuna
- Incorporate additional clinical or genetic data
- Build a web-based diagnostic tool using Streamlit or Flask
- Extend to **survival analysis** or **time-to-event modeling**
- Validate the model on an **external dataset**
