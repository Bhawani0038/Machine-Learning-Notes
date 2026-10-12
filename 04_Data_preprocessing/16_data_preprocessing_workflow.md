## Topic 16: Data Preprocessing Workflow and Best Practices

**Objective\:** Combine the preprocessing concepts we have learned into a practical workflow for machine learning.

Data preprocessing is not just one operation. It is a sequence of steps that converts raw data into a reliable format for model training.

## 1\. Complete preprocessing workflow

**1\. Load the Dataset**

Read CSV, Excel, database, or other data sources

**2\. Understand the Data**

Shape, data types, statistics, target distribution

**3\. Clean the Data**

Duplicates, invalid values, missing values, data types

**4\. Split the Data**

Separate training and testing data

**5\. Preprocess Features**

Imputation, encoding, scaling, transformations

**6\. Engineer and Select Features**

Create useful features and remove irrelevant ones

**7\. Handle Class Imbalance**

Use class weights or resampling when appropriate

**8\. Train and Evaluate**

Fit the model and evaluate on untouched test data

**Important\:** This is a general workflow, not a rigid sequence. For example, target-independent data inspection and cleaning can happen before splitting, but any operation that *learns parameters from the data*-such as imputing means, scaling, selecting features, or resampling-must be fitted using training data only. In cross-validation, fit these operations separately within each training fold.

## 2\. Practical preprocessing example

We will use the `clean_df` Customer Churn dataset from our previous lessons.

This example demonstrates\:

- Separating features and target.
- Splitting data.
- Imputing missing numerical values.
- Scaling numerical features.
- One-hot encoding categorical features.
- Training a logistic regression model.

Install the libraries if needed\:

```bash
pip install pandas scikit-learn
```

### Complete code

```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Separate features and target
X = clean_df.drop(
    columns=["customer_id", "churn", "churn_encoded"],
    errors="ignore"
)

y = clean_df["churn"].map({"No": 0, "Yes": 1})

# 2. Remove rows with an unknown target
valid_target = y.notna()
X = X.loc[valid_target]
y = y.loc[valid_target].astype(int)

# 3. Identify feature types
numeric_features = X.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# 4. Split before fitting learned preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Numerical preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 6. Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 7. Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 8. Build the complete workflow
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 9. Train
model.fit(X_train, y_train)

# 10. Predict
y_pred = model.predict(X_test)

# 11. Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
```

### Why use a pipeline?

A pipeline ensures that preprocessing steps are learned from the training data and then applied consistently to the test data.

It also reduces the risk of accidentally fitting the scaler, imputer, or encoder on the test set.

The dataset used in these lessons is small and illustrative. Its evaluation results should not be treated as reliable estimates of real-world customer churn performance.

## 3\. Final revision table

| Concept | Purpose |
|--------|--------|
| Missing-value imputation | Fill missing feature values |
| Duplicate removal | Remove repeated records when they are genuine duplicates |
| Data-type correction | Ensure columns have appropriate types |
| Outlier handling | Investigate and manage unusual observations |
| Encoding | Convert categorical features into model-compatible representations |
| Feature scaling | Put numerical features on comparable scales |
| Feature transformation | Change feature distributions or representations |
| Feature engineering | Create useful features |
| Feature selection | Remove irrelevant or redundant features |
| Leakage prevention | Prevent improper information from entering training |
| Imbalance handling | Improve minority-class learning when appropriate |