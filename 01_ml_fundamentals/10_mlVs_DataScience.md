# Module 1 — Machine Learning Fundamentals  
## Topic 10: Machine Learning vs Data Science

### What We Will Cover

- What is Data Science?
- What is Machine Learning?
- Relationship between Data Science and ML
- Data Science workflow
- ML workflow
- Key differences
- Skills required
- Real-world example
- Data Scientist vs ML Engineer
- When ML is actually needed

---

# 1. What is Data Science?

**Data Science is the broader process of using data, statistics, programming, analysis, and machine learning to extract useful insights and solve business problems.**

A Data Scientist may work on:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Data Analysis
      ↓
Visualization
      ↓
Statistics
      ↓
Machine Learning
      ↓
Business Insights
```

So, **Machine Learning is one important part of Data Science**, but Data Science is not limited to ML.

---

# 2. What is Machine Learning?

Machine Learning focuses on building systems that **learn patterns from data and make predictions or decisions**.

Example:

```text
Historical House Data
        ↓
Machine Learning
        ↓
House Price Model
        ↓
Predict price of new house
```

The main focus is:

> **Learning patterns from data to make predictions or decisions.**

---

# 3. Relationship Between Data Science and ML

A useful way to think about it:

```text
             DATA SCIENCE
        ┌───────────────────────┐
        │ Data Collection       │
        │ Data Cleaning         │
        │ SQL                   │
        │ Statistics            │
        │ EDA                   │
        │ Visualization         │
        │ Business Analysis     │
        │ Machine Learning      │
        │ Communication         │
        └───────────────────────┘
                    │
                    ↓
           Machine Learning
```

Machine Learning is **inside the broader Data Science ecosystem**.

However, ML is also a separate field used outside traditional Data Science workflows, such as production recommendation systems, computer vision, robotics, and large-scale prediction systems.

---

# 4. Data Science Example

Suppose an e-commerce company wants to understand why sales are decreasing.

A Data Scientist might:

### Step 1 — Collect Data

```text
Orders
Customers
Products
Website Activity
Marketing Data
```

### Step 2 — Clean Data

```text
Missing Values
Duplicates
Incorrect Data
Outliers
```

### Step 3 — Analyze

Questions:

- Which products are selling less?
- Which cities have declining sales?
- Which customer groups are leaving?
- Which marketing campaigns are performing poorly?

### Step 4 — Visualize

Create:

- Charts
- Dashboards
- Reports

### Step 5 — Machine Learning

Build a model to predict:

```text
Which customers are likely to stop purchasing?
```

### Step 6 — Business Recommendation

For example:

> Customers with declining purchase frequency and low recent engagement are more likely to churn.

This complete process is **Data Science**.

---

# 5. Machine Learning Example

Suppose the same company wants to predict customer churn.

Dataset:

| age | monthly_spend | orders | support_calls | churn |
|---:|---:|---:|---:|---|
| 25 | 1200 | 8 | 1 | No |
| 42 | 800 | 2 | 5 | Yes |
| 31 | 1500 | 10 | 0 | No |

The ML task is:

```text
Customer Data
      ↓
ML Algorithm
      ↓
Churn Model
      ↓
Yes / No
```

This is specifically a **Machine Learning task**.

---

# 6. Data Science vs Machine Learning

| Data Science | Machine Learning |
|---|---|
| Broader field | Specialized field |
| Focuses on extracting value from data | Focuses on learning patterns |
| Includes data analysis | Includes model training |
| Includes statistics | Uses statistics/math |
| Includes visualization | Usually focuses on prediction/decision |
| Includes SQL/data collection | Uses prepared data for modeling |
| Includes business understanding | Focuses strongly on model performance |
| Can work without ML | Requires a learning algorithm/model |

---

# 7. Skills Comparison

### Data Scientist

Common skills:

```text
Python
SQL
Statistics
Pandas
NumPy
Data Visualization
Power BI/Tableau
EDA
Machine Learning
Business Understanding
Communication
```

### Machine Learning Engineer

Common skills:

```text
Python
Machine Learning
Deep Learning
Scikit-learn
Model Deployment
APIs
Docker
Cloud
MLOps
Model Monitoring
Software Engineering
```

There is significant overlap between the roles.

---

# 8. Does Every Data Science Project Need ML?

**No.**

This is an important concept.

Suppose a company asks:

> "Which month had the highest sales?"

You don't need machine learning.

You can use:

```text
SQL
Pandas
Excel
Power BI
```

Similarly:

> "What was our average monthly revenue?"

Simple statistical analysis is enough.

ML becomes useful when the problem involves tasks such as:

- Prediction
- Classification
- Forecasting
- Pattern recognition
- Recommendation
- Anomaly detection
- Automated decisions

---

# 9. Simple Real-Life Comparison

Imagine a retail company.

### Data Science

```text
Why are sales decreasing?
Which products perform best?
Which city has highest revenue?
Who are our best customers?
```

These can be answered using:

- SQL
- Statistics
- EDA
- Visualization
- Business analysis

### Machine Learning

```text
What will next month's sales be?
Which customer will churn?
Which product should we recommend?
Will this transaction be fraudulent?
```

These are prediction/learning problems where ML may be appropriate.

---

# 10. Data Analyst vs Data Scientist vs ML Engineer

A simplified comparison:

| Role | Main Focus |
|---|---|
| Data Analyst | Understand what happened |
| Data Scientist | Understand data + build predictive models |
| ML Engineer | Build, deploy, and maintain ML systems |

### Example — Customer Churn

**Data Analyst:**

> "25% of customers churned last quarter."

**Data Scientist:**

> "Customers with certain behavioral patterns have a high probability of churning."

**ML Engineer:**

> "The churn model is deployed as an API and automatically generates predictions for new customers."

---

# 11. One Complete Picture

```text
                    DATA
                     ↓
              DATA SCIENCE
                     ↓
        ┌────────────┼────────────┐
        ↓            ↓            ↓
    Analysis     Statistics    Visualization
        │
        ↓
  Machine Learning
        ↓
     Prediction
        ↓
   Business Decision
```

And after the model is built:

```text
Model
  ↓
Deployment
  ↓
API / Application
  ↓
Real Users
  ↓
Monitoring
```

---

## Key Points

- **Data Science is broader than Machine Learning.**
- **Machine Learning is an important part of Data Science.**
- Data Science includes **SQL, statistics, EDA, visualization, business analysis, and ML**.
- ML mainly focuses on **learning patterns and making predictions/decisions**.
- Not every Data Science problem requires ML.
- Data Analyst → mainly analyzes data.
- Data Scientist → analyzes data and can build predictive models.
- ML Engineer → focuses strongly on building, deploying, and maintaining ML systems.
- The best solution is not always ML; **use ML when it provides value beyond simpler methods.**