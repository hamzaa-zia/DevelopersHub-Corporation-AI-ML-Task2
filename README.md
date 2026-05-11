# Heart Disease Risk Prediction

This project performs exploratory data analysis and builds classification models to predict heart disease risk from a Kaggle heart disease dataset. The workflow covers data cleaning, medical feature interpretation, EDA visualizations, Logistic Regression, Decision Tree classification, ROC-AUC evaluation, confusion matrices, and feature importance analysis.

---

## 📊 Dataset Overview

The dataset is stored in `heart.csv` and contains 499 patient records with 14 columns. The target column is `target`, where `1` represents patients with heart disease risk and `0` represents patients without heart disease risk.

The dataset has no missing values, but the script still includes a cleaning step so the workflow remains complete and reproducible.

---

## 🫀 Medical Features

| Column | Meaning |
|---|---|
| `age` | Patient age in years. |
| `sex` | Biological sex, commonly encoded as `1 = male` and `0 = female`. |
| `cp` | Chest pain type. Chest pain pattern is an important clinical signal for possible heart disease. |
| `trestbps` | Resting blood pressure in mm Hg. High blood pressure can increase heart workload. |
| `chol` | Serum cholesterol in mg/dL. High cholesterol can contribute to artery plaque buildup. |
| `fbs` | Fasting blood sugar above 120 mg/dL, encoded as a binary value. |
| `restecg` | Resting ECG result. ECG measures electrical activity of the heart. |
| `thalach` | Maximum heart rate achieved during exercise. Lower exercise capacity can indicate higher risk. |
| `exang` | Exercise-induced angina. Angina is chest pain caused by reduced blood flow to the heart. |
| `oldpeak` | ST depression caused by exercise compared with rest. Higher values may indicate abnormal heart response during stress. |
| `slope` | Slope of the peak exercise ST segment. It describes the ST segment pattern during stress testing. |
| `ca` | Number of major vessels colored by fluoroscopy. Vessel involvement can be linked with heart disease risk. |
| `thal` | Thalassemia or blood-flow test category used in the dataset. |
| `target` | Binary prediction label: `1 = heart disease risk`, `0 = no heart disease risk`. |

---

## 🧰 Tools and Libraries

- `pandas`: Loads the CSV file, inspects rows and columns, checks missing values, separates features from the target, and saves result tables.
- `matplotlib`: Creates and saves charts such as the target distribution, ROC curve, confusion matrix, and feature importance plots.
- `seaborn`: Builds clearer statistical visualizations, including count plots, histograms, and the correlation heatmap.
- `scikit-learn`: Handles preprocessing, train-test splitting, model training, prediction, and evaluation metrics.

Important `scikit-learn` functions used:

- `train_test_split`: Splits the dataset into training and testing data.
- `ColumnTransformer`: Applies different preprocessing steps to numeric and categorical columns.
- `StandardScaler`: Standardizes numeric features for Logistic Regression.
- `OneHotEncoder`: Converts categorical numeric codes into model-friendly encoded columns.
- `LogisticRegression`: Trains a probability-based binary classification model.
- `DecisionTreeClassifier`: Trains a rule-based classification model.
- `accuracy_score`: Calculates the percentage of correct predictions.
- `confusion_matrix` and `ConfusionMatrixDisplay`: Show correct and incorrect predictions by class.
- `roc_curve` and `roc_auc_score`: Measure how well the model separates risky and non-risky patients.
- `classification_report`: Reports precision, recall, and F1-score for both classes.

---

## 📚 Learning Resources

- Logistic Regression: learned from Gate Smashers.
- Decision Tree: learned from Gate Smashers.
- ROC-AUC and ROC Curve: learned from Nomiqo.

---

## 🔄 Workflow

1. Load `heart.csv` using `pandas`.
2. Check dataset shape, columns, data types, missing values, and target distribution.
3. Clean the dataset by filling numeric missing values with medians and categorical missing values with modes.
4. Generate EDA charts to understand feature trends and class balance.
5. Split the dataset into training and testing sets using stratified sampling.
6. Preprocess numeric and categorical features.
7. Train two classification models:
   - Logistic Regression
   - Decision Tree
8. Evaluate both models using accuracy, confusion matrix, ROC curve, and ROC-AUC.
9. Save model reports, plots, metrics, and feature importance outputs in the `outputs` folder.

---

## 📈 Model Results

The script trained and tested both models on the current dataset split.

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.94 | 0.983 |
| Decision Tree | 0.84 | 0.906 |

Logistic Regression performed better on this split. It achieved 94% accuracy and a ROC-AUC score of 0.983, which means it separated the two classes very well on the test data.

---

## 🧪 Confusion Matrix Interpretation

For Logistic Regression:

| Result | Count |
|---|---:|
| Correctly predicted no disease | 44 |
| Incorrectly predicted disease | 3 |
| Incorrectly missed disease | 3 |
| Correctly predicted disease | 50 |

In medical prediction, false negatives are especially important because they represent patients who may have heart disease risk but were predicted as no disease. Logistic Regression produced 3 false negatives on the test set.

---

## ⭐ Important Features

The strongest Logistic Regression signals included:

- `ca`: Number of major vessels.
- `cp`: Chest pain type.
- `thal`: Thalassemia or blood-flow test result.
- `oldpeak`: Exercise ST depression.

The strongest Decision Tree signals included:

- `cp`: Chest pain type.
- `ca`: Number of major vessels.
- `thal`: Thalassemia or blood-flow test result.
- `chol`: Serum cholesterol.
- `oldpeak`: Exercise ST depression.

These results suggest that chest pain type, vessel information, thalassemia-related test category, and stress-test response are major contributors to the prediction.

---

## 🗂️ Project Structure

```text
.
|-- heart.csv
|-- heart_disease_eda_model.py
|-- requirements.txt
|-- README.md
`-- outputs/
    |-- dataset_summary.txt
    |-- model_metrics.csv
    |-- target_distribution.png
    |-- medical_feature_reference.png
    |-- numeric_feature_distributions.png
    |-- categorical_feature_trends.png
    |-- correlation_heatmap.png
    |-- roc_curve_comparison.png
    |-- logistic_regression_confusion_matrix.png
    |-- decision_tree_confusion_matrix.png
    |-- logistic_regression_feature_importance.png
    `-- decision_tree_feature_importance.png
```

---

## ▶️ How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the complete analysis and modeling workflow:

```bash
python heart_disease_eda_model.py
```

After running the script, all reports and charts are saved in the `outputs` folder.

---

## 🖼️ Generated Outputs

- `dataset_summary.txt`: Dataset shape, columns, missing values, target distribution, and descriptive statistics.
- `target_distribution.png`: Shows the balance between heart disease and no heart disease cases.
- `medical_feature_reference.png`: Summarizes the medical dataset columns by patient profile, symptoms, clinical measurements, diagnostic tests, and target variable.
- `numeric_feature_distributions.png`: Shows how numeric features vary across target classes.
- `categorical_feature_trends.png`: Compares categorical medical features with the target.
- `correlation_heatmap.png`: Shows relationships between numeric columns.
- `roc_curve_comparison.png`: Compares model separation performance.
- `logistic_regression_confusion_matrix.png`: Shows Logistic Regression prediction errors.
- `decision_tree_confusion_matrix.png`: Shows Decision Tree prediction errors.
- `logistic_regression_feature_importance.png`: Shows top Logistic Regression feature signals.
- `decision_tree_feature_importance.png`: Shows top Decision Tree feature signals.

---

## 🔍 Chart Interpretation Guide

The PNG charts include labels, legends, or notes where possible. The details below explain what the numerical values mean.

| Chart | What the Numbers Represent |
|---|---|
| `target_distribution.png` | The bar labels show the number of patients in each target class. `0` means no heart disease risk and `1` means heart disease risk. |
| `medical_feature_reference.png` | Groups the dataset columns by medical meaning. It explains whether each feature represents patient profile, symptoms, clinical measurements, diagnostic test results, or the prediction target. |
| `numeric_feature_distributions.png` | The x-axis shows actual medical values such as age, cholesterol, resting blood pressure, maximum heart rate, or ST depression. The y-axis shows how many patients fall within each value range. |
| `categorical_feature_trends.png` | The x-axis shows encoded category values. For example, `sex` uses `0` and `1`, `exang` uses `0 = no` and `1 = yes`, and columns like `cp`, `restecg`, `slope`, `ca`, and `thal` use medical category codes from the dataset. |
| `correlation_heatmap.png` | Correlation values range from `-1` to `+1`. A positive value means two columns increase together, a negative value means one tends to decrease when the other increases, and values close to `0` show weak linear relationship. |
| `roc_curve_comparison.png` | The x-axis is false positive rate, meaning patients incorrectly predicted as having disease. The y-axis is true positive rate, meaning actual disease cases correctly detected. AUC closer to `1.0` is better. |
| `logistic_regression_confusion_matrix.png` | Each cell shows patient count. Diagonal cells are correct predictions, while off-diagonal cells are wrong predictions. |
| `decision_tree_confusion_matrix.png` | Each cell shows patient count for Decision Tree predictions. False negatives are medically important because they represent risky patients predicted as no disease. |
| `logistic_regression_feature_importance.png` | Longer bars mean stronger model influence. The values are absolute Logistic Regression coefficients, so they show strength of effect, not whether the feature increases or decreases risk. |
| `decision_tree_feature_importance.png` | Longer bars mean the Decision Tree used that feature more strongly for splitting patients into prediction groups. |

Encoded feature names in importance charts use the format `feature_value`. For example, `cp_0` means chest pain type category `0`, and `thal_3` means thalassemia test category `3`.

---

## ⚠️ Notes

This model is built for learning and analysis. It should not be used as a real medical diagnosis system. Real clinical prediction requires expert validation, larger patient samples, external testing, and careful handling of medical risk.
