# Machine Learning Notes — Final Course Structure

## Module 1 — Machine Learning Fundamentals

1. What is Machine Learning?
2. AI vs Machine Learning vs Deep Learning
3. Traditional Programming vs Machine Learning
4. Why Machine Learning?
5. Real-World Applications of ML
6. Types of Machine Learning
   - Supervised Learning
   - Unsupervised Learning
   - Semi-Supervised Learning
   - Reinforcement Learning
7. Machine Learning Terminology
   - Dataset
   - Feature
   - Target
   - Sample
   - Label
   - Model
   - Training
   - Prediction
   - Parameter
   - Hyperparameter
8. Classification vs Regression
9. Structured vs Unstructured Data
10. Machine Learning vs Data Science

---

# Module 2 — Mathematics & Statistics for ML

### Statistics
1. Population vs Sample
2. Mean_Median_Mode
3. Range_Variance_Standard Deviation
4. Percentiles & Quartiles_IQR_Outliers_Covariance_Correlation
5. Outliers
6. Covarance
7. Correlation

### Probability
13. Basic Probability
14. Events
15. Independent Events
16. Dependent Events
17. Conditional Probability
18. Bayes Theorem
19. Naive Bayes Intuition

### Distributions
20. Normal Distribution
21. Standard Normal Distribution
22. Z-Score
23. Central Limit Theorem

### Hypothesis Testing
24. Null & Alternative Hypothesis
25. Type I & Type II Errors
26. p-value
27. Confidence Interval
28. Basic Hypothesis Testing

### Mathematics
29. Vectors
30. Matrices
31. Dot Product
32. Linear Equations
33. Functions
34. Derivatives
35. Gradient
36. Gradient Descent

---

# Module 3 — Complete ML Workflow



1. Problem Definition
2. Business Understanding
3. Data Collection
4. Data Understanding
5. Data Cleaning
6. Exploratory Data Analysis
7. Feature Engineering
8. Feature Selection
9. Train-Test Split
10. Model Selection
11. Model Training
12. Prediction
13. Model Evaluation
14. Hyperparameter Tuning
15. Final Model
16. Model Saving
17. Deployment
18. Monitoring


---

# Module 4 — Data Preprocessing

1. Understanding Dataset
2. Numerical Features
3. Categorical Features
4. Missing Values
5. Duplicate Values
6. Incorrect Data Types
7. Outliers
8. Data Cleaning
9. Encoding
   - Label Encoding
   - One-Hot Encoding
10. Feature Scaling
   - Standardization
   - Normalization
11. Feature Transformation
12. Feature Selection
13. Feature Engineering
14. Data Leakage
15. Train-Test Split
16. Preprocessing Pipeline
17. `Pipeline`
18. `ColumnTransformer`

---

# Module 5 — Regression


1. What is Regression?
2. Regression vs Classification
3. Linear Regression
4. Simple Linear Regression
5. Multiple Linear Regression
6. Cost Function
7. Mean Squared Error
8. Gradient Descent
9. Linear Regression using Scikit-learn
10. Polynomial Regression
11. Polynomial Features
12. Ridge Regression
13. Lasso Regression
14. Elastic Net
15. Regularization
16. Regression Model Comparison

### Practical Projects

- House Price Prediction
- Salary Prediction
- Sales Prediction

---

# Module 6 — Classification

1. What is Classification?
2. Binary Classification
3. Multiclass Classification
4. Multilabel Classification
5. Logistic Regression
6. Probability & Classification
7. Decision Boundary
8. K-Nearest Neighbors
9. Naive Bayes
10. Decision Tree
11. Random Forest
12. Support Vector Machine
13. Gradient Boosting
14. XGBoost

### Practical Projects

- Student Pass/Fail
- Customer Churn
- Loan Approval
- Disease Prediction

---

# Module 7 — Model Evaluation

This module should be taught **after students understand regression and classification**, because then the metrics have practical meaning.

### Regression

1. MAE
2. MSE
3. RMSE
4. R²
5. Adjusted R²
6. Cross Validation

### Classification

6. Confusion Matrix
7. TP
8. TN
9. FP
10. FN
11. Accuracy
12. Precision
13. Recall
14. F1 Score
15. Classification Report
16. ROC Curve
17. AUC
18. Threshold
19. Precision-Recall Trade-off
20. Cross Validation

### Practical Decision Making

20. Accuracy vs Precision vs Recall
21. False Positive Cost
22. False Negative Cost
23. Choosing the Right Metric

---

# Module 8 — Unsupervised Learning

1. What is Unsupervised Learning?
2. Supervised vs Unsupervised Learning
3. Clustering
4. K-Means
5. K-Means Algorithm
6. Centroids
7. Distance Calculation
8. Choosing K
9. Elbow Method
10. Silhouette Score
11. Hierarchical Clustering
12. Dendrogram
13. DBSCAN
14. Dimensionality Reduction
15. PCA
16. PCA Intuition
17. PCA in Python


### Project

**Customer Segmentation**

```text
Customer Data
      ↓
Preprocessing
      ↓
K-Means
      ↓
Customer Groups
      ↓
Business Interpretation
```

---

# Module 9 — Model Improvement

1. Underfitting
2. Overfitting
3. Bias
4. Variance
5. Bias-Variance Tradeoff
6. Cross Validation
7. K-Fold Cross Validation
8. Stratified K-Fold
9. Hyperparameters
10. Hyperparameter Tuning
11. GridSearchCV
12. RandomizedSearchCV
13. Regularization
14. Feature Selection
15. Handling Imbalanced Data
16. Class Weights
17. SMOTE
18. Data Leakage Prevention

---

# Module 10 — Ensemble Learning

1. What is Ensemble Learning?
2. Why Ensemble Models?
3. Voting
4. Bagging
5. Random Forest
6. Boosting
7. AdaBoost
8. Gradient Boosting
9. XGBoost
10. Stacking
11. Random Forest vs Gradient Boosting
12. Ensemble Model Comparison

---

# Module 11 — End-to-End ML Projects

Rather than teaching projects as isolated tutorials, each project should follow the **same professional workflow**.

### Project 1 — House Price Prediction

Regression

### Project 2 — Student Result Prediction

Classification

### Project 3 — Customer Churn Prediction

Classification

### Project 4 — Loan Approval Prediction

Classification

### Project 5 — Customer Segmentation

Clustering

### Project 6 — Sales Prediction

Regression

### Project 7 — Employee Salary Prediction

Regression

### Project 8 — Medical Diagnosis

Classification + Evaluation

Each project:

```text
Business Problem
       ↓
Dataset
       ↓
EDA
       ↓
Preprocessing
       ↓
Feature Engineering
       ↓
Train-Test Split
       ↓
Model
       ↓
Evaluation
       ↓
Tuning
       ↓
Final Model
```

---

# Module 12 — ML Deployment

1. What is Model Deployment?
2. Saving ML Models
3. `joblib`
4. Loading Models
5. Building Prediction Application
6. Streamlit ML Application
7. FastAPI ML API
8. Connecting Frontend to ML API
9. Requirements File
10. Virtual Environment
11. Docker Basics
12. Deployment Concepts
13. Model Versioning
14. Basic ML Monitoring

