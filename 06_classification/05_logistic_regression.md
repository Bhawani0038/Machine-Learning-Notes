## Topic 5: Logistic Regression

### 1\. Definition

**Logistic Regression is a supervised machine learning algorithm primarily used for classification.** It estimates the probability that an observation belongs to a particular class.

Despite its name, Logistic Regression is commonly used to predict categories such as\:

- Pass or Fail
- Spam or Not Spam
- Loan Approved or Rejected
- Customer Churn or Not Churn
- Fraudulent or Genuine Transaction

For binary classification, the model estimates the probability of class `1`, then uses a decision threshold to select the predicted class.

## 2\. Understand the intuition

Suppose we want to predict whether a student will pass based on study hours.

| Study hours | Result |
|--------|--------|
| 1 | Fail |
| 2 | Fail |
| 3 | Fail |
| 4 | Fail |
| 5 | Pass |
| 6 | Pass |
| 7 | Pass |
| 8 | Pass |

We want a model that learns the relationship between study hours and the probability of passing.

For a new student studying 5.5 hours, the model might estimate\:

$$
P(\text{Pass})=0.75
$$

If the decision threshold is 0.5\:

$$
0.75 \geq 0.5 \implies \text{Pass}
$$

The probability is an estimate, not a guarantee that the student will pass.

## 3\. Why not use Linear Regression?

Linear Regression predicts numerical values using\:

$$
\hat{y}=wx+b
$$

Here, $w$ is the coefficient and $b$ is the intercept.

For classification, this creates a problem: the output can be less than 0 or greater than 1, so it cannot always represent a valid probability.

For example, a linear model might predict\:

- `-0.2` - invalid as a probability.
- `1.3` - also invalid as a probability.

Logistic Regression addresses this by applying the **sigmoid function** to the linear score.

## 4\. The sigmoid function

The sigmoid function converts any real-valued input into a number strictly between 0 and 1.

$$
\boxed{\sigma(z)=\frac{1}{1+e^{-z}}}
$$

Where\:

- $z$ is the model's linear score.
- $e$ is Euler's number, approximately 2.71828.
- $\sigma(z)$ is the estimated probability of class 1.

For a single feature\:

$$
z=wx+b
$$

Therefore, Logistic Regression predicts\:

$$
\boxed{P(y=1\mid x)=\frac{1}{1+e^{-(wx+b)}}}
$$


## 5\. Mathematical example

Suppose the trained model has\:

$$
z=1.2x-5
$$

For a student who studies 5 hours\:

**Step 1: Calculate the linear score.**

$$
z=1.2(5)-5=1
$$

**Step 2: Apply the sigmoid function.**

$$
P(y=1\mid x)=\frac{1}{1+e^{-1}}
$$

$$
P(y=1\mid x)\approx 0.7311
$$

**Step 3: Apply the decision threshold.**

Since $0.7311 \geq 0.5$, the predicted class is `1` \(Pass).

## 6\. Python implementation from scratch

This example demonstrates the prediction mathematics without using Scikit-learn.

```python
import math

def predict_probability(hours, weight, bias):
    z = weight * hours + bias
    return 1 / (1 + math.exp(-z))

weight = 1.2
bias = -5

hours = 5

probability = predict_probability(hours, weight, bias)
prediction = 1 if probability >= 0.5 else 0

print(f"Pass probability: {probability:.4f}")
print("Predicted result:", "Pass" if prediction == 1 else "Fail")
```

Output\:

```text
Pass probability: 0.7311
Predicted result: Pass
```

Here, the weight and bias are manually specified for demonstration. We have not trained a model yet.

## 7\. Train Logistic Regression using Scikit-learn

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.DataFrame({
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "result":      [0, 0, 0, 0, 1, 1, 1, 1]
})

X = df[["study_hours"]]
y = df["result"]

model = LogisticRegression()
model.fit(X, y)

new_student = pd.DataFrame({"study_hours": [5]})

probabilities = model.predict_proba(new_student)
prediction = model.predict(new_student)

print("Class probabilities:", probabilities.round(3))
print("Predicted class:", prediction[0])
print("Weight:", model.coef_[0][0])
print("Bias:", model.intercept_[0])
```

Unlike our manual example, Scikit-learn learns the coefficient and intercept from the training data.

## 8\. How does Logistic Regression learn?

The model must find suitable coefficients and an intercept from labeled examples.

For binary classification, it typically minimizes **log loss**, also called binary cross-entropy\:

$$
J=-\frac{1}{m}\sum_{i=1}^{m}
\left[y_i\log(p_i)+(1-y_i)\log(1-p_i)\right]
$$

Where\:

- $m$ \= number of training observations.
- $y_i$ \= actual class, either 0 or 1.
- $p_i$ \= predicted probability of class 1.

Predictions that assign very low probability to the correct class incur a large penalty. The optimizer adjusts the coefficients to reduce the loss, subject to any regularization applied.

## 9\. Key concepts to remember

| Concept | Purpose |
|--------|--------|
| Linear score | Combines features and coefficients |
| Sigmoid | Converts the score into a probability |
| Log loss | Measures the quality of probability predictions |
| Coefficients | Represent learned feature effects on log-odds |
| Intercept | Provides the baseline linear score |
| Threshold | Converts probability into a class label |
