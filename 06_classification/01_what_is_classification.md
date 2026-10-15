# Topic 1 - What is Classification?

## 1\. Definition

**Classification is a supervised machine learning technique used to predict a category or class from input data.**

The output is a category rather than a continuous numerical value.

Examples\:

| Problem | Input | Predicted class |
|--------|--------|--------|
| Email detection | Email content | Spam / Not Spam |
| Loan approval | Income, credit score | Approved / Rejected |
| Student result | Marks, attendance | Pass / Fail |
| Customer churn | Usage, complaints | Churn / Not Churn |
| Image recognition | Image pixels | Cat / Dog / Bird |

## 2\. Classification vs Regression

| Classification | Regression |
|--------|--------|
| Predicts categories | Predicts numerical values |
| Spam or Not Spam | House price |
| Pass or Fail | Student marks |
| Fraud or Genuine | Monthly sales |
| Logistic Regression, Decision Tree Classifier | Linear Regression, Random Forest Regressor |

Remember: the word *regression* in Logistic Regression does not mean that its primary task is predicting continuous values. It is used for classification.

## 3\. How classification works

**1\. Collect labeled data**

Study hours, attendance, previous marks and result

**2\. Train a classification model**

The model learns patterns from examples

**3\. Provide new input**

A new student's study hours and attendance

**4\. Predict a class**

Pass

## 4\. Practical Python example

We will use a realistic, small student dataset to understand the concept before progressing to more advanced algorithms.

```python
import pandas as pd

# Training data
df = pd.DataFrame({
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "attendance":  [40, 50, 55, 60, 70, 75, 85, 90],
    "result":      [0, 0, 0, 0, 1, 1, 1, 1]
})

print(df)
```

Here, `result` is our target variable\:

- `0` \= Fail
- `1` \= Pass

The other two columns are features that help the model make its prediction.

Train a Logistic Regression classifier\:

```python
from sklearn.linear_model import LogisticRegression

# Features and target
X = df[["study_hours", "attendance"]]
y = df["result"]

# Create and train the model
model = LogisticRegression()
model.fit(X, y)

# Predict for a new student
new_student = pd.DataFrame({
    "study_hours": [5.5],
    "attendance": [72]
})

prediction = model.predict(new_student)

print("Predicted result:",
      "Pass" if prediction[0] == 1 else "Fail")
```
