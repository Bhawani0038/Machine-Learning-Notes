# Module 1 — Machine Learning Fundamentals  
## Topic 7: Machine Learning Terminology

### What We Will Cover

- Dataset
- Sample / Data Point / Observation
- Feature
- Target
- Label
- `X` and `y`
- Algorithm
- Model
- Training
- Prediction / Inference
- Parameters
- Hyperparameters
- Training Data
- Validation Data
- Test Data

---

## 1. Dataset

A **dataset** is a collection of data used to analyze a problem or train a machine learning model.

Example — House Price Dataset:

| sqft_area | rooms | parking | price |
|---:|---:|---:|---:|
| 1000 | 2 | 1 | 45 |
| 1500 | 3 | 1 | 65 |
| 2000 | 4 | 2 | 90 |

Here, the complete table is the **dataset**.

---

## 2. Sample / Data Point / Observation

Each individual row in a dataset is called a:

- Sample
- Data Point
- Observation
- Record
- Row

For example:

```text
1500, 3, 1, 65
```

represents **one house**, so it is one sample.

---

## 3. Feature

A **feature** is an input variable used by the model to make a prediction.

For the house-price dataset:

```text
sqft_area
rooms
parking
```

are features.

### Example

If we want to predict house price:

```text
sqft_area + rooms + parking → price
```

The model uses the features to learn the relationship with the target.

---

## 4. Target

The **target** is the value that we want the model to predict.

In our example:

```text
price
```

is the target.

Other names for target:

- Output
- Dependent Variable
- Response Variable
- Target Variable

---

## 5. Label

The word **label** is commonly used for the known answer associated with a training example, especially in classification.

Example:

| study_hours | attendance | result |
|---:|---:|---|
| 2 | 65 | Fail |
| 5 | 80 | Pass |
| 7 | 90 | Pass |

Here:

```text
study_hours
attendance
```

→ Features

```text
result
```

→ Target / Label

For classification, we commonly say **class label**.

---

# 6. X and y

In Python/Scikit-learn, a common convention is:

```python
X = features
y = target
```

Example:

```python
X = df[["sqft_area", "rooms", "parking"]]

y = df["price"]
```

So:

```text
X → Input Features
y → Target / Output
```

### Simple Diagram

```text
             X
     ┌─────────────────┐
     │ sqft_area       │
     │ rooms           │
     │ parking         │
     └────────┬────────┘
              ↓
        ML Model
              ↓
             y
           price
```

---

# 7. Algorithm

An **algorithm** is the learning method used to learn patterns from data.

Examples:

- Linear Regression
- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM
- K-Means

For example:

```text
Dataset
   ↓
Linear Regression Algorithm
   ↓
Trained Model
```

---

# 8. Model

A **model** is the learned result after an algorithm has been trained on data.

For example:

```text
Training Data
      ↓
Linear Regression Algorithm
      ↓
Trained Linear Regression Model
```

The trained model can then make predictions on new data.

### Important Difference

| Algorithm | Model |
|---|---|
| Learning method | Learned result |
| Used to learn patterns | Contains learned patterns |
| Example: Linear Regression | Trained Linear Regression |
| Exists before training | Created after training |

---

# 9. Training

**Training** means providing data to an ML algorithm so that it can learn patterns.

```text
Training Data
     ↓
Algorithm
     ↓
Learning
     ↓
Model
```

In Scikit-learn, training commonly happens with:

```python
model.fit(X_train, y_train)
```

---

# 10. Prediction / Inference

After training, we provide new data to the model and ask it for an output.

This is called:

- Prediction
- Inference

Example:

```python
prediction = model.predict(X_test)
```

For house prices:

```text
New House
1500 sqft
3 rooms
1 parking
       ↓
    Model
       ↓
Predicted Price
```

---

# 11. Parameters

**Parameters are values learned by the model during training.**

For Linear Regression:

```text
price = m₁ × sqft + m₂ × rooms + m₃ × parking + b
```

The model learns:

```text
m₁
m₂
m₃
b
```

These are **model parameters**.

The important point is:

> Parameters are learned from the training data.

---

# 12. Hyperparameters

**Hyperparameters are settings chosen before or during model training.**

They are not normally learned directly from the training data.

Examples:

### Random Forest

```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=10
)
```

Here:

```text
n_estimators → Hyperparameter
max_depth    → Hyperparameter
```

### KNN

```python
KNeighborsClassifier(n_neighbors=5)
```

```text
n_neighbors → Hyperparameter
```

### Difference

| Parameter | Hyperparameter |
|---|---|
| Learned during training | Set by the user/algorithm configuration |
| Part of trained model | Controls how training happens |
| Example: Linear Regression coefficients | Example: `max_depth` |
| Learned from data | Selected before/during training |

---

# 13. Training Data

Data used to **teach the model** is called training data.

```text
Dataset
   ↓
Training Data
   ↓
Model learns
```

---

# 14. Validation Data

Validation data is used to:

- Compare models
- Tune hyperparameters
- Select the best model
- Check model performance during development

Example:

```text
Training Data → Learn
Validation Data → Choose/Tune
Test Data → Final Evaluation
```

---

# 15. Test Data

Test data is used to evaluate the final model on data that was not used for training.

```text
Training → Learn
Validation → Tune
Test → Final Evaluation
```

A common split might be:

```text
80% → Training
20% → Testing
```

Later, we will study validation and cross-validation in detail.

---

# Complete House Price Example

Suppose we have:

| sqft_area | rooms | parking | price |
|---:|---:|---:|---:|
| 1000 | 2 | 1 | 45 |
| 1500 | 3 | 1 | 65 |
| 2000 | 4 | 2 | 90 |

### Terminology

```text
Dataset
   ↓
3 rows
   ↓
Each row = Sample

Features (X)
   ├── sqft_area
   ├── rooms
   └── parking

Target (y)
   └── price

Algorithm
   └── Linear Regression

Training
   └── model.fit()

Model
   └── Trained Linear Regression

Prediction
   └── model.predict()
```

---

## One-Line Revision

| Term | Meaning |
|---|---|
| Dataset | Complete collection of data |
| Sample | One row/data point |
| Feature | Input variable |
| Target | Value to predict |
| Label | Known output/class |
| `X` | Input features |
| `y` | Target |
| Algorithm | Learning method |
| Model | Trained learning result |
| Training | Learning from data |
| Prediction | Output for new data |
| Parameter | Learned by model |
| Hyperparameter | Model setting |
| Training Data | Data used for learning |
| Validation Data | Data used for tuning/selection |
| Test Data | Data used for final evaluation |

### Key Points

- **Features → inputs**
- **Target → output**
- **`X` → features**
- **`y` → target**
- **Algorithm → learning method**
- **Model → trained result**
- **Parameters → learned values**
- **Hyperparameters → selected settings**
- **Training → model learns**
- **Prediction → model gives output**
- **Test data → final performance check**