# Heart Disease Prediction Using Machine Learning

This repository contains a complete machine learning workflow for predicting heart disease risk using the UCI Cleveland dataset. The analysis is built for academic reporting and includes:
- dataset profiling and cleaning,
- multi-class and binary classification,
- model evaluation with confusion matrices and class-wise metrics,
- feature-importance based clinical interpretation,
- actionable improvement recommendations for future work.

## 1. Objective
The main objective is to build and evaluate a clinically meaningful heart disease prediction model.

Two prediction settings are analyzed:
1. Multi-class diagnosis (`num` = 0, 1, 2, 3, 4)
2. Binary diagnosis (`0 = Healthy`, `1 = Heart Disease Presence`)

The binary setting is emphasized for screening use, where minimizing false negatives is critical.

## 2. Repository Contents
- `heart_disease.csv`: Source dataset
- `prabhath_analysis.ipynb`: Main end-to-end analysis notebook
- `prabhath.ipynb`: Additional notebook
- `output.png`, `output2.png`, `output3.png`: Saved visual outputs
- `README.md`: Project documentation

## 3. Dataset Summary
Dataset: UCI Heart Disease (Cleveland subset)

Common attributes used by the model include:
- `age`: Age in years
- `sex`: Sex (1 = male, 0 = female)
- `cp`: Chest pain type
- `trestbps`: Resting blood pressure (mm Hg)
- `chol`: Serum cholesterol (mg/dl)
- `fbs`: Fasting blood sugar > 120 mg/dl
- `restecg`: Resting ECG category
- `thalach`: Maximum heart rate achieved
- `exang`: Exercise-induced angina
- `oldpeak`: ST depression induced by exercise
- `slope`: Slope of peak exercise ST segment
- `ca`: Number of major vessels colored by fluoroscopy
- `thal`: Thalassemia category
- `num`: Target variable (0 healthy, 1-4 disease stages)

## 4. Workflow and Methodology
### 4.1 Data Cleaning
- Loaded the CSV with pandas
- Replaced hidden missing markers (`?`) with nulls
- Removed incomplete rows to keep training input consistent

### 4.2 Preprocessing
- Split features and target
- Used train/test split (`test_size=0.2`, `random_state=42`)
- Applied `StandardScaler` to standardize numeric features

### 4.3 Baseline Multi-class Modeling
- Trained Random Forest for multi-class diagnosis
- Compared multiple values of `n_estimators` (`50, 100, 150, 200`)
- Selected and evaluated final model

### 4.4 Optimized Binary Modeling
- Collapsed disease stages (`1-4`) into a single positive class
- Re-trained Random Forest using same split strategy
- Evaluated with:
1. Accuracy
2. Classification report (precision, recall, F1-score)
3. Confusion matrix

### 4.5 Clinical Interpretation
- Computed feature importance from Random Forest
- Ranked top predictors and linked them to clinical factors
- Highlighted practical risk: false negatives (disease predicted as healthy)

## 5. Results (What to Report)
The notebook is designed to produce report-ready outputs:
- Multi-class performance summary
- Binary performance summary
- Confusion matrices for both settings
- Feature-importance visualization
- Recommendation table for model improvement

In the binary scenario, the analysis prioritizes recall for the disease class to reduce missed diagnoses.

## 6. Recommendations Included in Notebook
The notebook already includes a structured improvement plan:
- tune for higher disease recall,
- run hyperparameter optimization (GridSearchCV),
- use stratified cross-validation,
- address class imbalance,
- compare with additional baselines (KNN, Logistic Regression),
- improve explainability (for example, permutation importance / SHAP-style analysis).

## 7. Reproducibility Guide
### 7.1 Environment
Use Python 3.9+ (recommended: 3.10 or newer).

### 7.2 Install dependencies
```bash
pip install pandas numpy scikit-learn seaborn matplotlib ipython jupyter
```

### 7.3 Run analysis
1. Open `prabhath_analysis.ipynb`
2. Run cells from top to bottom in order
3. Review printed metrics, confusion matrices, and feature-importance charts

## 8. Important Notes
- This project supports academic analysis and learning; it is not a production medical system.
- Results can vary slightly across environments and library versions.
- For clinical deployment, additional validation, calibration, fairness checks, and regulatory review are required.

## 9. Academic Context
This work is structured to align with common machine learning assignment criteria:
- dataset understanding,
- preprocessing quality,
- implementation rigor,
- evaluation depth,
- critical discussion and improvement planning.