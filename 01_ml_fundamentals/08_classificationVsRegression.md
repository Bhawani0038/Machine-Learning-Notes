# Module 1 — Machine Learning Fundamentals  
## Topic 8: Classification vs Regression

### What We Will Cover

- What is Regression?
- What is Classification?
- Numerical vs Categorical Target
- Regression Examples
- Classification Examples
- Binary Classification
- Multiclass Classification
- Regression vs Classification
- How to identify the problem type
- Real-world examples
- Key Points

---

# 1. What is Regression?

**Regression is a supervised machine learning problem where the target is a continuous numerical value.**

The model predicts a **number**.

### Examples

| Problem | Target |
|---|---:|
| House price prediction | ₹75 lakh |
| Salary prediction | ₹65,000 |
| Sales prediction | ₹2,50,000 |
| Temperature prediction | 32.5°C |
| Delivery time prediction | 45.7 minutes |

Example:

```text
House Features
      ↓
ML Model
      ↓
₹82.5 Lakh
```

The output can be a different numerical value.

---

# 2. What is Classification?

**Classification is a supervised machine learning problem where the target represents a category or class.**

The model predicts a **class/category**.

### Examples

| Problem | Target |
|---|---|
| Student result | Pass / Fail |
| Email detection | Spam / Not Spam |
| Loan approval | Approved / Rejected |
| Disease detection | Disease / No Disease |
| Customer churn | Yes / No |

Example:

```text
Student Data
      ↓
ML Model
      ↓
Pass
```

---

# 3. Main Difference

The easiest way to identify them:

```text
Is the target a numerical continuous value?
            ↓
           YES
            ↓
        REGRESSION
```

```text
Is the target a category/class?
            ↓
           YES
            ↓
       CLASSIFICATION
```

---

# 4. Regression Example — House Price

Suppose:

| sqft_area | rooms | price |
|---:|---:|---:|
| 1000 | 2 | 45 |
| 1500 | 3 | 65 |
| 2000 | 4 | 90 |
| 2500 | 4 | 115 |

We want to predict:

```text
sqft_area = 1800
rooms = 3
```

The model might predict:

```text
₹78.5 lakh
```

Since the target is a numerical value:

**This is Regression.**

---

# 5. Classification Example — Student Result

Suppose:

| study_hours | attendance | result |
|---:|---:|---|
| 2 | 60 | Fail |
| 4 | 75 | Pass |
| 6 | 85 | Pass |
| 1 | 55 | Fail |

We want to predict the result of a new student.

```text
study_hours = 5
attendance = 80
```

Model output:

```text
Pass
```

Since the target is a category:

**This is Classification.**

---

# 6. Binary Classification

When there are **two possible classes**, it is called **Binary Classification**.

Examples:

```text
Pass / Fail
Yes / No
Spam / Not Spam
Fraud / Not Fraud
Churn / Not Churn
Approved / Rejected
```

Example:

```text
Customer Data
      ↓
Classification Model
      ↓
     Churn
```

---

# 7. Multiclass Classification

When there are **more than two classes**, it is called **Multiclass Classification**.

Example:

```text
Customer Complaint
       ↓
     Model
       ↓
 ┌───────────────┐
 │ Billing       │
 │ Technical     │
 │ Delivery      │
 │ Refund        │
 └───────────────┘
```

Other examples:

- Image → Cat / Dog / Horse
- Disease → Flu / COVID / Allergy
- Product → Electronics / Clothing / Grocery
- Student grade → A / B / C / D

---

# 8. Regression vs Classification

| Feature | Regression | Classification |
|---|---|---|
| Target | Numerical | Categorical |
| Output | Number | Class |
| Example | Price | Pass/Fail |
| Example | Salary | Churn/No Churn |
| Example | Sales | Spam/Not Spam |
| Common Algorithms | Linear Regression | Logistic Regression |
| Common Algorithms | Decision Tree Regressor | Decision Tree Classifier |
| Common Algorithms | Random Forest Regressor | Random Forest Classifier |
| Evaluation | MAE, MSE, RMSE, R² | Accuracy, Precision, Recall, F1 |

---

# 9. Important: Classification Can Produce Probabilities

Suppose a classification model predicts whether a customer will churn.

It may internally produce:

```text
Probability of Churn = 0.82
```

Then based on a threshold:

```text
0.82 ≥ 0.50
       ↓
     Churn
```

So classification can involve numerical probabilities, but the **final target is a category**.

---

# 10. How to Identify the Problem

Always look at the **target column**.

### Example 1

```text
Target = price
```

Possible values:

```text
45.5
52.8
67.2
81.4
```

→ **Regression**

### Example 2

```text
Target = result
```

Possible values:

```text
Pass
Fail
```

→ **Classification**

### Example 3

```text
Target = salary
```

Possible values:

```text
35000
42000
55000
72000
```

→ **Regression**

### Example 4

```text
Target = loan_status
```

Possible values:

```text
Approved
Rejected
```

→ **Classification**

---

# 11. Common Algorithms

### Regression

Some commonly used regression algorithms:

```text
Linear Regression
Polynomial Regression
Ridge Regression
Lasso Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
```

### Classification

Common classification algorithms:

```text
Logistic Regression
KNN
Naive Bayes
Decision Tree Classifier
Random Forest Classifier
SVM
Gradient Boosting
XGBoost
```

We will study these algorithms in detail in later modules.

---

# 12. Simple Decision Rule

```text
             Target
                │
        ┌───────┴────────┐
        │                │
    Numerical         Category
        │                │
        ↓                ↓
   REGRESSION       CLASSIFICATION
        │                │
     Price          Pass / Fail
     Salary         Spam / Ham
     Sales          Yes / No
```

---

## Key Points

- **Regression → predicts a numerical value.**
- **Classification → predicts a category/class.**
- `price`, `salary`, `sales` → usually Regression.
- `Pass/Fail`, `Yes/No`, `Spam/Not Spam` → Classification.
- **Binary Classification → 2 classes.**
- **Multiclass Classification → more than 2 classes.**
- Always determine the ML problem type by looking at the **target variable**.