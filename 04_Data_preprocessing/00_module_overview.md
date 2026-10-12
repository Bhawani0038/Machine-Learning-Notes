# Module 4 - Data Preprocessing

## 1\. Dataset: Customer Churn Prediction

We will use these columns throughout the module.

| Column | Data type | Purpose |
|--------|--------|--------|
| `customer_id` | Categorical/ID | Identify each customer |
| `age` | Numerical | Customer's age |
| `monthly_charges` | Numerical | Monthly bill |
| `tenure_months` | Numerical | Months with the company |
| `support_calls` | Numerical | Number of support calls |
| `gender` | Categorical | Male/Female |
| `contract_type` | Categorical | Month-to-month, One year, Two year |
| `internet_service` | Categorical | Fiber, DSL, None |
| `payment_method` | Categorical | UPI, Credit Card, Bank Transfer |
| `churn` | Target | Yes/No |

**Target variable\:** `churn`

- `Yes` - customer left the company.
- `No` - customer stayed with the company.

This dataset is useful because we can apply both classification algorithms and preprocessing techniques without changing the business problem.

> Quick note for students: This module is concept-heavy, but you do not need to memorize every alternative method for every dataset. The important workflow is: inspect -> clean -> split -> fit preprocessing on training data only -> evaluate.

## 2\. Create a practice dataset with preprocessing problems

We will deliberately include missing values, duplicate records, incorrect data types, categorical columns, and outliers so students can practice cleaning real-world data.

This dataset is intentionally small for learning. Later, you can scale the same structure to hundreds or thousands of records for more realistic model evaluation.

## 3\. How we will apply all 18 topics

### Common mistakes students should avoid

- Fitting imputers, scalers, encoders, or feature selectors on the full dataset before splitting.
- Treating every outlier as an error without checking whether the value is valid.
- Using `customer_id` as a model feature without a clear reason.
- Applying `LabelEncoder` to every categorical feature instead of using a problem-aware encoding strategy.
- Ignoring class imbalance and relying only on accuracy.
- Creating new features using information that would not be available at prediction time.

These are the main errors that make a preprocessing notebook look correct but produce poor real-world models.

| No. | Topic | What we will do |
|--------|--------|--------|
| 1 | Understanding Dataset | `head()`, `info()`, `describe()` |
| 2 | Numerical Features | Analyze age, charges, tenure, calls |
| 3 | Categorical Features | Inspect unique values and category counts |
| 4 | Missing Values | Impute missing age, charges, and gender |
| 5 | Duplicate Values | Detect and remove duplicate records |
| 6 | Incorrect Data Types | Convert tenure to numeric |
| 7 | Outliers | Detect unusual ages, charges, and calls |
| 8 | Data Cleaning | Correct inconsistent categories and invalid values |
| 9 | Encoding | Label encode the target; one-hot encode nominal features |
| 10 | Feature Scaling | Apply standardization and normalization |
| 11 | Feature Transformation | Use transformations such as `log1p()` for skewed data |
| 12 | Feature Selection | Remove identifiers and irrelevant features |
| 13 | Feature Engineering | Create features such as `average_monthly_charge` where meaningful |
| 14 | Data Leakage | Keep target-derived information out of input features |
| 15 | Handling Imbalanced Data | Use class weights or resampling when needed |
| 16 | Data Preprocessing Workflow | Combine cleaning, split, preprocessing, and evaluation |
| 17 | `Pipeline` | Chain imputers, scalers, encoders, and a model |
| 18 | `ColumnTransformer` | Apply different preprocessing to numerical and categorical columns |