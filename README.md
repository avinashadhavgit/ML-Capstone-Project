# Employee Attrition Prediction — ML Capstone Project

## Project Overview
Employee attrition poses significant operational and financial challenges for organizations, resulting in high recruitment costs, loss of institutional knowledge, and reduced team productivity. This project implements an end-to-end Machine Learning pipeline to predict whether an employee is likely to leave the organization (`Attrition: Yes/No`), identify key underlying churn factors, and provide actionable intelligence for proactive HR intervention.

---

##  Business Objectives
* **Predict Attrition Risk:** Build a supervised learning model to identify employees at high risk of departure.
* **Identify Key Attrition Drivers:** Determine critical workload, compensation, and satisfaction metrics impacting retention.
* **Support Strategic HR Decisions:** Enable HR leadership to implement targeted, data-driven retention programs before employees resign.

---

## Dataset & Features Overview
The model utilizes the ** HR Employee Attrition Dataset**, comprising **1,470 employee records** and **35 initial attributes**.

### Key Input Features:
* **Demographics:** `Age`, `Gender`, `MaritalStatus`, `Education`, `EducationField`
* **Job & Work Profile:** `Department`, `JobRole`, `JobLevel`, `BusinessTravel`, `DistanceFromHome`, `OverTime`
* **Compensation & Rates:** `MonthlyIncome`, `DailyRate`, `HourlyRate`, `MonthlyRate`, `PercentSalaryHike`, `StockOptionLevel`
* **Satisfaction & Ratings (1–4 Scale):** `EnvironmentSatisfaction`, `JobSatisfaction`, `JobInvolvement`, `RelationshipSatisfaction`, `WorkLifeBalance`
* **Tenure & Work History:** `TotalWorkingYears`, `NumCompaniesWorked`, `YearsAtCompany`, `YearsInCurrentRole`, `YearsSinceLastPromotion`, `YearsWithCurrManager`, `TrainingTimesLastYear`, `PerformanceRating`

---

##  Project Pipeline & Methodology

### 1. Data Cleaning & Preprocessing
* **Removal of Uninformative Features:** Constant and unique identifier columns (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`) were dropped to prevent noise and dimensionality bloat.
* **Outlier Treatment:** Implemented custom IQR-based capping (`cap_outlier` using `series.clip(LB, UB)`) across all numerical features:
  	Lower Bound = Q1 - 1.5 *IQR , 
        Upper Bound = Q3 + 1.5* IQR
* **Categorical Encoding:** Converted non-numeric text attributes into numeric form using `LabelEncoder`.

### 2. Exploratory Data Analysis (EDA)
* Generated univariate histograms (`df.hist()`) to inspect feature distributions.
* Evaluated inter-feature correlations using a Seaborn correlation heatmap (`sns.heatmap(corr)`).

### 3. Feature Scaling & Data Splitting
* Split the dataset into **80% Training Set** (`x_train`) and **20% Testing Set** (`x_test`).
* Standardized features using `StandardScaler` to ensure zero mean and unit variance across models.

### 4. Multi-Model Cross-Validation & Evaluation
Evaluated 5 distinct classifiers using **5-Fold Stratified Cross-Validation** (`StratifiedKFold`) scored on Accuracy, Precision, Recall, and F1-Score:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Gaussian Naive Bayes**
5. **K-Nearest Neighbors (KNN)**

---

##  Model Evaluation & Results

### 5-Fold Stratified Cross-Validation Summary:

| Model | CV Accuracy | CV Precision | CV Recall | CV F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Gaussian Naive Bayes** | **77.6%** | **40.1%** | **63.1%** | **0.489** |
| **Logistic Regression** | 86.4% | 67.6% | 37.3% | 0.480 |
| **Decision Tree** | 78.7% | 37.1% | 37.9% | 0.374 |
| **Random Forest** | 85.2% | 71.5% | 20.2% | 0.314 |
| **K-Nearest Neighbors** | 84.1% | 62.9% | 14.7% | 0.237 |

###  Model Selection Rationale
**Gaussian Naive Bayes** was selected as the final production model. In employee turnover prediction, **Recall** is the most critical business metric—failing to flag an actual resigning employee (False Negative) is significantly costlier than reviewing a retained employee (False Positive). Gaussian Naive Bayes achieved the highest **Recall (63.1%)** and top **F1-Score (0.489)**.

---

##  Inference on New Employee Data
The script includes an inline inference demonstration for newly onboarded employee profiles:
```python
# Predict Attrition status for a sample employee record
prediction = best_model.predict(new_df)
result = le.inverse_transform(prediction)

if prediction[0] == 1:
    print("There will be Attrition")
else:
    print("There will be no Attrition")
```

---

##  Key Business Insights & HR Recommendations
* **Manage Overtime Exhaustion:** Overtime is the single highest predictor of turnover. Implement strict workload limits in frontline roles.
* **Review Entry-Level Compensation:** Employees earning lower monthly salaries and residing farther from work show higher churn. Competitive pay band adjustments reduce flight risk.
* **Focus on Early Tenure Engagement:** Flight risk is highest during the first 1–3 years at the organization and under new management; structured mentorship programs should be prioritized.
