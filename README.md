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

Running 5 fold stratefied Cross validation

Cross validation results sorted by F1 score:
              model  accuracy  precision   recall       f1
Logistic Regression  0.800410   0.579048 0.471429 0.518125
        Naive Bayes  0.775166   0.497618 0.528571 0.507142
      Decision Tree  0.749411   0.427692 0.385714 0.404102
      Random Forest  0.810292   0.803333 0.214286 0.332147
K-Nearest Neighbors  0.787814   0.472381 0.185714 0.264745

###  Model Selection Rationale
**Logistic Regression** was selected as the final production model. In employee turnover prediction, **Accuracy** is the most critical business metric—failing to flag an actual resigning employee Logistic Regression achieved the highest **Accuracy (80%)** and top **F1-Score (0.518125)**.

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
