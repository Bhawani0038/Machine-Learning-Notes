# Module 1 — Machine Learning Fundamentals

## Topic 6: Types of Machine Learning

### What We Will Cover

- Supervised Learning
- Unsupervised Learning
- Semi-Supervised Learning
- Reinforcement Learning
- Main differences
- Real-world examples
- Regression
- Classification
- Clustering
- How to identify the type of ML problem

---

# 1. Overview

Machine Learning can be broadly divided into four major types:

```text
Machine Learning
│
├── Supervised Learning
│
├── Unsupervised Learning
│
├── Semi-Supervised Learning
│
└── Reinforcement Learning
```

The biggest difference is **how the model learns from data and whether correct answers are provided during training**.

---

# 2. Supervised Learning

In **Supervised Learning**, the training data contains:

- Input features
- Known target/output

The model learns the relationship between them.

```text
Input Features + Known Target
             ↓
       ML Algorithm
             ↓
           Model
             ↓
        New Input
             ↓
        Prediction
```

### Example

Suppose we have:

| Study Hours | Result |
|---:|---|
| 1 | Fail |
| 2 | Fail |
| 3 | Fail |
| 4 | Pass |
| 5 | Pass |
| 6 | Pass |

Here:

```text
Feature → Study Hours
Target  → Result
```

The target is already known in the training data.

Therefore, this is **Supervised Learning**.

---

# 3. Two Major Types of Supervised Learning

```text
Supervised Learning
│
├── Regression
│
└── Classification
```

---

## 3.1 Regression

Regression is used when the target is a **continuous numerical value**.

### Examples

```text
House Price → ₹75.5 Lakh
Salary → ₹65,000
Temperature → 32.5°C
Sales → ₹1,25,000
```

Example:

| Area | Price |
|---:|---:|
| 850 | 32 |
| 1100 | 45 |
| 1350 | 55 |
| 1600 | 68 |

The model learns:

```text
Area → Price
```

and predicts the price of a new house.

### Common Regression Algorithms

- Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Ridge
- Lasso
- Elastic Net
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

---

# 4. Classification

Classification is used when the target represents a **category/class**.

Examples:

```text
Spam / Not Spam
Pass / Fail
Fraud / Normal
Yes / No
Disease / No Disease
```

Example:

| Study Hours | Result |
|---:|---|
| 1 | Fail |
| 2 | Fail |
| 4 | Pass |
| 5 | Pass |

The model learns:

```text
Study Hours → Result
```

and predicts a class for a new student.

### Common Classification Algorithms

- Logistic Regression
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- Random Forest
- Support Vector Machine
- Gradient Boosting
- XGBoost

---

# 5. Binary Classification

When there are **two classes**, it is called binary classification.

Examples:

```text
Spam / Not Spam
Pass / Fail
Fraud / Not Fraud
Churn / Not Churn
```

Example:

```text
Customer
   ↓
ML Model
   ↓
Churn = Yes
```

---

# 6. Multiclass Classification

When there are **more than two classes**, it is called multiclass classification.

Example:

```text
Email
  ↓
Model
  ↓
Business / Promotion / Social / Spam
```

Another example:

```text
Image
  ↓
Model
  ↓
Cat / Dog / Horse / Car
```

There are more than two possible classes.

---

# 7. Unsupervised Learning

In **Unsupervised Learning**, the training data does not contain a known target/label for each observation.

```text
Input Data
    ↓
ML Algorithm
    ↓
Patterns / Groups
```

The model attempts to discover useful structure in the data.

---

# 8. Example: Customer Segmentation

Suppose an e-commerce company has:

| Customer | Spending | Purchases |
|---|---:|---:|
| A | 90,000 | 25 |
| B | 85,000 | 22 |
| C | 15,000 | 4 |
| D | 12,000 | 3 |
| E | 50,000 | 12 |

There is no predefined column saying:

```text
Customer Type
```

We ask the algorithm to identify groups.

```text
Customer Data
      ↓
   K-Means
      ↓
Customer Groups
```

It might discover groups such as:

```text
Group 1 → High-value customers
Group 2 → Medium-value customers
Group 3 → Low-value customers
```

This is **Clustering**, an unsupervised learning task.

---

# 9. Common Unsupervised Learning Tasks

### Clustering

Find groups in data.

Examples:

- Customer segmentation
- Market segmentation

Algorithms:

- K-Means
- Hierarchical Clustering
- DBSCAN

### Dimensionality Reduction

Reduce the number of features while attempting to preserve useful information.

Examples:

- PCA
- Other dimensionality-reduction techniques

Uses include:

- Visualization
- Noise reduction
- Feature compression
- Preprocessing

---

# 10. Semi-Supervised Learning

Sometimes we have:

```text
Small amount of labeled data
+
Large amount of unlabeled data
```

This is the basic idea behind **Semi-Supervised Learning**.

Example:

```text
10,000 Images
│
├── 500 labeled images
│
└── 9,500 unlabeled images
```

Instead of ignoring the unlabeled images, a semi-supervised approach can use information from both labeled and unlabeled data.

```text
Labeled Data
     +
Unlabeled Data
     ↓
Learning Algorithm
     ↓
Model
```

This can be useful when obtaining labels is expensive or time-consuming.

---

# 11. Reinforcement Learning

**Reinforcement Learning (RL)** is based on an **agent interacting with an environment** and learning through rewards and penalties.

Basic structure:

```text
        Environment
             ↑
             │
          Action
             │
           Agent
             │
             ↓
          Reward
```

More completely:

```text
Agent
  │
  │ Action
  ↓
Environment
  │
  ├── New State
  │
  └── Reward
        ↓
      Agent
```

The agent learns which actions tend to produce better long-term outcomes.

---

# 12. Example: Game Playing

Suppose an agent is learning to play a game.

```text
Game State
    ↓
  Agent
    ↓
 Action
    ↓
 Game
    ↓
 Reward
```

For example:

```text
Win  → +10
Lose → -10
```

The agent tries different actions and gradually learns a strategy that maximizes cumulative reward.

Reinforcement learning is commonly associated with problems involving **sequential decisions and interaction with an environment**.

---

# 13. Real-Life Examples of Each Type

| Type | Example | Typical Task |
|---|---|---|
| Supervised | House Price | Regression |
| Supervised | Spam Detection | Classification |
| Supervised | Loan Default | Classification |
| Unsupervised | Customer Segmentation | Clustering |
| Unsupervised | Grouping Products | Clustering |
| Unsupervised | Feature Reduction | PCA |
| Semi-Supervised | Image Classification with Few Labels | Classification |
| Reinforcement | Game Playing | Sequential Decision Making |
| Reinforcement | Robot Control | Sequential Decision Making |

---

# 14. The Most Important Difference

Students should remember this simple rule:

### Supervised

**Answers are available during training.**

```text
Data + Answer
     ↓
   Model
```

### Unsupervised

**Answers are not provided.**

```text
Data
 ↓
Model
 ↓
Patterns / Groups
```

### Semi-Supervised

**Some answers are available, many are not.**

```text
Labeled Data + Unlabeled Data
              ↓
            Model
```

### Reinforcement

**The system learns from actions and rewards.**

```text
Action
  ↓
Environment
  ↓
Reward
  ↓
Learning
```

---

# 15. Quick Decision Framework

When you receive a new ML problem, ask:

### Question 1: Do I have a target?

```text
Yes → Supervised Learning
No  → Unsupervised Learning may be appropriate
```

### Question 2: If I have a target, what type is it?

```text
Continuous number
        ↓
    Regression
```

```text
Category
   ↓
Classification
```

### Question 3: If I don't have a target, what do I want?

```text
Groups
  ↓
Clustering
```

```text
Fewer dimensions/features
        ↓
Dimensionality Reduction
```

### Question 4: Is the system learning through interaction and rewards?

```text
Yes → Reinforcement Learning
```

---

# 16. Complete Picture

```text
                         MACHINE LEARNING
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
     SUPERVISED           UNSUPERVISED        REINFORCEMENT
          │                     │                     │
      ┌───┴───┐             ┌───┴────┐            Agent
      │       │             │        │               ↓
 Regression Classification Clustering  PCA        Action
                                                      ↓
                                                   Reward
```

And:

```text
             SUPERVISED
                 │
        ┌────────┴────────┐
        ↓                 ↓
   Regression       Classification
        │                 │
   Price            Spam/Not Spam
   Salary           Pass/Fail
   Sales            Fraud/Normal
```

---

## Key Points

- **Supervised Learning** learns from labeled data.
- **Regression** predicts continuous numerical values.
- **Classification** predicts categories/classes.
- **Unsupervised Learning** works without predefined target labels.
- **Clustering** discovers groups in data.
- **Dimensionality Reduction** reduces feature dimensions.
- **Semi-Supervised Learning** uses both labeled and unlabeled data.
- **Reinforcement Learning** learns through actions, states, and rewards.
- The presence or absence of a **target** is one of the first questions to ask when identifying an ML problem.

