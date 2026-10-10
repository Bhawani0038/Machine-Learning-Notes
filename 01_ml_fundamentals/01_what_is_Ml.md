# Module 1 — Machine Learning Fundamentals

## Topic 1: What is Machine Learning?

### What We Will Cover

- Machine Learning definition
- Traditional Programming vs Machine Learning
- Basic ML idea
- Real-life examples
- How ML learns from data
- Features and Target
- Training and Prediction
- Simple ML workflow
- Key terms

---

## 1. What is Machine Learning?

**Machine Learning (ML)** is a branch of Artificial Intelligence that allows computers to **learn patterns from data and use those patterns to make predictions or decisions**.

Instead of explicitly programming every rule, we provide the computer with **data and an appropriate learning algorithm**, and the algorithm learns a model from that data.

For example:

> Instead of manually writing rules to determine whether a student will pass, we can provide previous students' study hours and results. The ML model can learn the relationship and predict the result for a new student.

---

## 2. Traditional Programming vs Machine Learning

### Traditional Programming

We explicitly provide the rules.

```text
Input Data + Rules
       ↓
    Program
       ↓
     Output
```

Example:

```python
marks = 75

if marks >= 40:
    result = "Pass"
else:
    result = "Fail"

print(result)
```

Here, **we wrote the rule**:

```text
marks >= 40 → Pass
marks < 40  → Fail
```

---

### Machine Learning

In ML, we provide examples and allow the algorithm to learn a relationship.

```text
Input Data + Correct Answers
             ↓
        ML Algorithm
             ↓
           Model
             ↓
      New Input Data
             ↓
        Prediction
```

Example training data:

| Study Hours | Result |
|---:|---|
| 1 | Fail |
| 2 | Fail |
| 3 | Fail |
| 4 | Pass |
| 5 | Pass |
| 6 | Pass |

The ML algorithm can learn the relationship between **study hours** and **result**.

Then:

```text
New Student
Study Hours = 4.5
       ↓
     Model
       ↓
Prediction = Pass
```

The model has learned a pattern from previous examples rather than us manually writing a `4-hour` rule.

---

## 3. Basic Idea of Machine Learning

The fundamental idea is:

```text
Historical Data
      ↓
Learning Algorithm
      ↓
Trained Model
      ↓
New Data
      ↓
Prediction / Decision
```

The model attempts to learn a useful relationship between the input data and the desired output.

In supervised learning, the data contains both **features (`X`)** and a **target (`y`)** that the model learns to predict. [Scikit-learn](https://scikit-learn.org/1.4/tutorial/basic/tutorial.html?utm_source=chatgpt.com)

---

## 4. Example: House Price Prediction

Suppose we have:

| Area (sqft) | Rooms | Price (₹ Lakh) |
|---:|---:|---:|
| 850 | 2 | 32 |
| 1100 | 2 | 45 |
| 1350 | 3 | 55 |
| 1600 | 3 | 68 |
| 2000 | 4 | 85 |
| 2400 | 4 | 105 |

Here:

**Features (`X`)**

```text
Area
Rooms
```

**Target (`y`)**

```text
Price
```

The ML algorithm learns:

```text
Area + Rooms
      ↓
   ML Model
      ↓
Predicted Price
```

For a new house:

```text
Area = 1800 sqft
Rooms = 3
```

the model might predict:

```text
Price = ₹75 Lakh
```

This is a **regression problem** because the target is a continuous numerical value. Scikit-learn similarly defines regression as predicting a continuous-valued target. [Scikit-learn](https://scikit-learn.com/?utm_source=chatgpt.com)

---

## 5. How Does Machine Learning Learn?

A simplified view:

```text
                    Training Data
                         ↓
                ┌─────────────────┐
                │ ML Algorithm    │
                └─────────────────┘
                         ↓
                   Learned Model
                         ↓
                ┌─────────────────┐
New Data ──────→│     Model       │
                └─────────────────┘
                         ↓
                    Prediction
```

The model identifies patterns or relationships in the training data.

For example:

```text
More House Area
       ↓
Usually Higher Price
```

or:

```text
More Study Hours
       ↓
Higher Probability of Passing
```

The exact relationship depends on the dataset and the algorithm.

---

## 6. Important ML Terms

| Term | Meaning |
|---|---|
| **Dataset** | Collection of data used for ML |
| **Sample / Row** | One individual observation |
| **Feature** | Input variable used by the model |
| **Target** | Output the model tries to predict |
| **Label** | Another common term for the target, especially in classification |
| **Model** | Learned mathematical representation of patterns in data |
| **Training** | Process of learning from training data |
| **Prediction** | Output produced by a trained model |
| **Parameter** | Value learned by the model during training |
| **Hyperparameter** | Setting chosen before/during training to control the learning process |

In scikit-learn terminology, features may also be called attributes, predictors, or input variables. [GitHub](https://github.com/scikit-learn/scikit-learn/wiki/Glossary?utm_source=chatgpt.com)

---

## 7. A Simple ML Example

Consider:

```text
Study Hours → Result
```

Training data:

```text
X = Study Hours

y = Result
```

For example:

```text
X = [1, 2, 3, 4, 5, 6]

y = [Fail, Fail, Fail, Pass, Pass, Pass]
```

The model learns from:

```text
X → y
```

After training:

```text
Study Hours = 4.5
       ↓
     Model
       ↓
Prediction = Pass
```

In Python/Scikit-learn, the general estimator workflow is based around fitting a model with training data and then using it for prediction. [Scikit-learn](https://scikit-learn.org/stable/getting_started?utm_source=chatgpt.com)

---

## 8. Where is Machine Learning Used?

| Application | ML Task |
|---|---|
| House Price Prediction | Regression |
| Spam Detection | Classification |
| Customer Churn | Classification |
| Product Recommendation | Recommendation |
| Customer Segmentation | Clustering |
| Fraud Detection | Classification / Anomaly Detection |
| Sales Forecasting | Regression / Time Series |
| Image Recognition | Classification |
| Credit Risk Prediction | Classification |
| Search Ranking | Ranking |

Scikit-learn's current documentation groups common ML capabilities into areas such as classification, regression, clustering, dimensionality reduction, preprocessing, and model selection. [Scikit-learn](https://scikit-learn.com/?utm_source=chatgpt.com)

---

## 9. The Core ML Idea

Remember this:

```text
                 DATA
                  ↓
             LEARNING
                  ↓
                MODEL
                  ↓
             NEW DATA
                  ↓
             PREDICTION
```

### One-line definition

> **Machine Learning is the process of learning useful patterns from data so that a computer can make predictions or decisions on new data.**

### Key Points

- ML learns from **data**.
- ML models learn **patterns/relationships**.
- **Features** are inputs.
- **Target** is the output we want to predict.
- **Training** means learning from data.
- **Prediction** means applying the learned model to new data.
- Regression predicts **continuous values**.
- Classification predicts **categories/classes**.
- Unsupervised learning works without a supplied target. [Scikit-learn](https://scikit-learn.org/1.4/tutorial/basic/tutorial.html?utm_source=chatgpt.com)

