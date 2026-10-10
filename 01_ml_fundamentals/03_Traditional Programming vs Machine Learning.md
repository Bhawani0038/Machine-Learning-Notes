# Module 1 — Machine Learning Fundamentals

## Topic 3: Traditional Programming vs Machine Learning

### What We Will Cover

- Traditional Programming
- Machine Learning approach
- Main difference
- Input, rules, output
- Training data
- Prediction
- Real-life examples
- When to use each approach
- Key terminology

---

# 1. Traditional Programming

In **traditional programming**, a programmer explicitly defines the rules or logic that the computer must follow.

```text
Input Data + Rules
       ↓
    Program
       ↓
     Output
```

### Example: Pass/Fail

Suppose the rule is:

```text
Marks >= 40 → Pass
Marks < 40  → Fail
```

Python:

```python
marks = 65

if marks >= 40:
    result = "Pass"
else:
    result = "Fail"

print(result)
```

Output:

```text
Pass
```

Here, the computer does not learn anything.

**We provide the rule.**

---

# 2. Machine Learning

In Machine Learning, instead of explicitly writing all the rules, we provide **examples/data**, and the algorithm learns patterns from that data.

```text
Data + Expected Output
          ↓
    ML Algorithm
          ↓
         Model
          ↓
      New Data
          ↓
      Prediction
```

### Example

Suppose we have historical student data:

| Study Hours | Result |
|---:|---|
| 1 | Fail |
| 2 | Fail |
| 3 | Fail |
| 4 | Pass |
| 5 | Pass |
| 6 | Pass |

We provide this data to an ML algorithm.

The algorithm learns a relationship between:

```text
Study Hours → Result
```

Then we give:

```text
Study Hours = 4.5
```

The trained model might predict:

```text
Pass
```

We didn't explicitly write:

```text
IF study_hours >= 4 THEN Pass
```

The model learned a pattern from the training examples.

---

# 3. The Main Difference

The biggest difference is **where the rules come from**.

### Traditional Programming

```text
Human
  ↓
Writes Rules
  ↓
Program
  ↓
Output
```

### Machine Learning

```text
Human
  ↓
Provides Data
  ↓
ML Algorithm
  ↓
Learns Patterns
  ↓
Model
  ↓
Prediction
```

---

# 4. Side-by-Side Comparison

| Traditional Programming | Machine Learning |
|---|---|
| Programmer writes rules | Model learns patterns |
| Rules are explicit | Rules are learned |
| Input + Program → Output | Input + Output examples → Model |
| Suitable for well-defined rules | Suitable when rules are difficult to define |
| Usually deterministic for fixed inputs | Predictions depend on learned model |
| No training phase required | Training phase required |
| Example: Tax calculation | Example: Spam detection |
| Example: Calculator | Example: House price prediction |
| Example: Payroll rules | Example: Customer churn prediction |

---

# 5. Example: Spam Detection

Consider an email:

> "Congratulations! You have won a free prize."

### Traditional Programming

We could manually create rules:

```text
IF email contains "free"
OR email contains "winner"
OR email contains "prize"
THEN Spam
```

But real-world spam is much more complicated.

Spammers can write:

```text
"You have been selected for a special reward."
```

The manually written rules may fail.

---

### Machine Learning

We can provide thousands of historical emails:

| Email | Category |
|---|---|
| Win a free prize | Spam |
| Claim your reward | Spam |
| Meeting at 3 PM | Not Spam |
| Project report attached | Not Spam |
| Special offer for you | Spam |

The ML model learns patterns from these examples.

```text
Historical Emails
       ↓
ML Algorithm
       ↓
Spam Detection Model
       ↓
New Email
       ↓
Spam / Not Spam
```

This is a situation where machine learning can be much more practical than manually maintaining a huge collection of rules.

---

# 6. Example: House Price Prediction

Suppose we want to predict house prices.

### Traditional Programming

We might try to write rules:

```text
IF area > 2000
    price = ₹80 lakh

IF area > 2500
    price = ₹100 lakh
```

But house prices depend on many factors:

```text
Area
Rooms
Location
Parking
Floor
Age
Distance from railway station
Amenities
...
```

Writing rules for every combination becomes difficult.

---

### Machine Learning

Instead, we provide historical house data:

| Area | Rooms | Parking | Price |
|---:|---:|---:|---:|
| 850 | 2 | 1 | 32 |
| 1100 | 2 | 1 | 45 |
| 1350 | 3 | 1 | 55 |
| 1600 | 3 | 1 | 68 |
| 2000 | 4 | 1 | 85 |

The model learns relationships between the features and price.

```text
Historical House Data
          ↓
     ML Algorithm
          ↓
      ML Model
          ↓
New House Information
          ↓
   Predicted Price
```

---

# 7. Traditional Programming Has Rules

A useful way to think about traditional programming:

```text
We know the rules
       ↓
Write the rules
       ↓
Computer follows them
```

Examples:

### Calculator

```text
a + b
```

The addition rule is already known.

### Tax Calculation

If the tax rules are explicitly defined:

```text
income
  ↓
tax rules
  ↓
tax amount
```

### ATM

The banking system follows predefined business rules for:

- Authentication
- Balance checking
- Withdrawal limits
- Transaction processing

These don't necessarily require machine learning.

---

# 8. Machine Learning Is Useful When Rules Are Difficult

A useful question is:

> **Can we easily write the rules ourselves?**

### If YES

Traditional programming may be appropriate.

Examples:

```text
Calculate salary
Calculate tax
Calculate interest
Validate an email format
Calculate shopping cart total
```

### If NO

But we have enough historical data:

Machine Learning may be appropriate.

Examples:

```text
Is this email spam?
Will this customer leave?
What will this house cost?
Is this transaction suspicious?
Will this student pass?
```

---

# 9. Important Difference in Data Flow

### Traditional Programming

```text
             Rules
               ↓
Input ─────→ PROGRAM ─────→ Output
```

### Machine Learning

During training:

```text
Input Data ─────┐
                ├──→ ML Algorithm ──→ Model
Expected Output ┘
```

During prediction:

```text
New Input ─────→ Trained Model ─────→ Prediction
```

This distinction is extremely important.

---

# 10. Training vs Prediction

### Training

The model learns from historical data.

```text
Training Data
     ↓
Algorithm
     ↓
Model
```

### Prediction

The trained model processes new data.

```text
New Data
    ↓
Trained Model
    ↓
Prediction
```

So:

> **Training = Learning**

> **Prediction = Using what was learned**

---

# 11. A Real-World Example: Customer Churn

Suppose a telecom company wants to predict whether a customer will leave.

Historical data:

| Tenure | Monthly Bill | Complaints | Churn |
|---:|---:|---:|---|
| 2 | 900 | 5 | Yes |
| 24 | 500 | 0 | No |
| 5 | 850 | 4 | Yes |
| 36 | 450 | 1 | No |

### Traditional Approach

The company could manually create rules:

```text
IF complaints > 3
AND tenure < 6
THEN customer will leave
```

But real customer behavior can involve many interacting factors.

### ML Approach

Provide historical customer data and outcomes:

```text
Customer Data
      ↓
ML Algorithm
      ↓
Churn Model
      ↓
New Customer
      ↓
Churn Probability
```

For example:

```text
Customer Churn Probability = 0.82
```

The business can then decide how to respond.

---

# 12. Important Terminology

| Term | Meaning |
|---|---|
| **Input** | Information given to the system |
| **Rule** | Explicit logic written by a programmer |
| **Feature** | Input variable used by an ML model |
| **Target** | Desired output during supervised learning |
| **Algorithm** | Procedure used to learn from data |
| **Model** | Learned representation produced through training |
| **Training** | Learning from training data |
| **Prediction** | Model output for new data |

---

# 13. The Core Difference

Remember this diagram:

```text
TRADITIONAL PROGRAMMING

       DATA + RULES
            ↓
         PROGRAM
            ↓
          OUTPUT
```

```text
MACHINE LEARNING

    DATA + EXPECTED OUTPUT
             ↓
        ML ALGORITHM
             ↓
           MODEL
             ↓
         NEW DATA
             ↓
        PREDICTION
```

