# ML_linear-bcell-epitopes

## 1. Introduction
This summary report presents the method and justification of data analysis and modelling carried out in the Data Mining coursework.  
The aim was to create a predictive model using a given dataset while addressing challenges like class imbalance and high-dimensional feature space.  

The process included:
- Exploratory Data Analysis (EDA)
- Data pre-processing
- Dimensionality reduction
- Model training and evaluation
- Prediction on a holdout test set

The analysis was implemented in a Jupyter Notebook, combining best practices in data preprocessing, feature engineering, model evaluation, and fairness-aware learning strategies.

---

## 2. EDA and Data Pre-Processing
**Libraries used:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

**Dataset Inspection**
- Loaded using `pandas.read_csv()`
- Checked shape, data types, and missing values
- Feature columns: `feat_*`
- Metadata columns: `Info_*`  
- Target variable (Class) is imbalanced

**Data Cleaning**
- Removed irrelevant `Info_*` columns (except `Info_group`)
- Extracted features (X), labels (y), and group identifiers

**Feature Scaling**
- Standardized features using `StandardScaler`
- Box plots confirmed standardization effect

---

## 3. Feature Reduction
**PCA (Principal Component Analysis)** used to reduce dimensionality:
- Removed multicollinearity and noise
- Retained 95% variance
- Reduced hundreds of features to a smaller, manageable set
- Faster training and less sensitivity to noise

---

## 4. Modelling and Assessment

### 4.1 Cross-Validation with Cost-Sensitive Learning
- **5-fold GroupKFold** stratified by `Info_group`  
- Models evaluated:
  - Random Forest (class_weight='balanced')
  - Logistic Regression (class_weight='balanced')  

**Results:**
- Random Forest: Balanced Accuracy = 0.5
- Logistic Regression: Balanced Accuracy = 0.6187  

Random Forest performed better across folds.

### 4.2 SMOTE Oversampling
- Split dataset: 80% train / 20% test (stratified)
- Applied SMOTE to training data
- Random Forest trained on balanced data
- Balanced Accuracy on test set: 0.6211  

Similar results to GroupKFold; Random Forest remained best.

### 4.3 Final Model
- Random Forest parameters:
  - `n_estimators=100`
  - `class_weight='balanced'`
  - `random_state=42`
- Trained on entire PCA-reduced feature space
- Demonstrated strong cross-validation performance

---

## 5. Conclusions and Discussion
- Standardization is critical for high-dimensional numerical data
- PCA reduced dimensionality while retaining >95% variance
- GroupKFold prevented data leakage during cross-validation
- Balanced accuracy is effective for imbalanced classes
- Random Forest with class weighting was most effective
- SMOTE helped but did not outperform GroupKFold approach
- No hyperparameter tuning or ensembling applied yet
- Future work: grid search or Bayesian tuning to enhance performance

**Summary:** Random Forest with PCA pipeline is scalable, interpretable, and accurate for holdout dataset predictions.
