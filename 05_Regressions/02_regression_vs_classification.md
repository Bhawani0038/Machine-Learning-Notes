# 2\. Regression vs Classification

## 1\. What is the Difference?

Both Regression and Classification are types of **Supervised Machine Learning**\. They learn from labeled training data, but they predict different types of outputs.

- **Regression\:** Predicts a continuous numerical value.
- **Classification\:** Predicts a class or category.

## 2\. Real-Life Examples

**Regression - House Price Prediction**

Input: Area = 1,500 sq. ft., Rooms = 3

Output: ₹65 lakh

The model predicts a numerical price.

---


**Classification - Student Result Prediction**

Input: Study hours = 2, Attendance = 40%

Output: Fail

The model predicts a category.

## 3\. Comparison Table

| Basis | Regression | Classification |
|--------|--------|--------|
| Purpose | Predict numerical values | Predict categories |
| Output | Continuous number | Class label |
| Example | Salary = ₹50,000 | Salary class = High |
| Target variable | Numerical | Categorical |
| Algorithms | Linear Regression, Ridge | Logistic Regression, SVM |
| Evaluation metrics | MAE, MSE, RMSE, $R^2$ | Accuracy, Precision, Recall, F1-score |
| Business use | Sales forecasting | Fraud detection |

## 4\. Python Implementation

We will use Scikit-learn to understand both approaches.

### A. Regression Example

**Problem\:** Predict student marks from study hours.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Input features: study hours
X = np.array([[1], [2], [3], [4], [5]])

# Target: marks
y = np.array([20, 30, 40, 50, 60])

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[6]])

print("Predicted Marks:", prediction[0])
```

Output\:

```text
Predicted Marks: 70.0
```

The model predicts a numerical value of `70.0`\.

### B. Classification Example

**Problem\:** Predict whether a student passes or fails based on study hours.

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Input features: study hours
X = np.array([[1], [2], [3], [4], [5], [6]])

# Target: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1])

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

prediction = model.predict([[5]])

result = "Pass" if prediction[0] == 1 else "Fail"

print("Predicted Result:", result)
```

Output\:

```text
Predicted Result: Pass
```

The model predicts the class `Pass`, represented internally by `1`\.

*Note\:* These small datasets are designed for learning. Real-world models require more representative data and proper evaluation.

## 5\. Can the Same Problem Use Both Techniques?

Yes. The choice depends on what you want to predict.

| Business Problem | Regression Output | Classification Output |
|--------|--------|--------|
| Student performance | Marks = 85 | Result = Pass |
| Customer churn | Expected months until churn | Churn = Yes/No |
| Loan application | Expected loan amount | Approved/Rejected |
| Product sales | Sales = ₹50,000 | Sales category = High/Low |