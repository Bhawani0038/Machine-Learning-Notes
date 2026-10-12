# Module 2 — Mathematics & Statistics for ML

## Topic 21: Statistical Tests — Z-Test, T-Test & Chi-Square Test

### What We Will Cover
- Why Different Statistical Tests?
- Z-Test
- T-Test
- Chi-Square Test
- When to Use Each Test
- Practical Examples
- Test Selection
- ML / Data Science Applications
- Key Points

---

# 1. Why Do We Need Statistical Tests?

Hypothesis testing gives us a framework for deciding whether an observed difference or relationship is statistically significant.

Different types of data and questions require different tests.

```text id="9d9j0p"
Question
   ↓
What type of data?
   ↓
What are we comparing?
   ↓
Choose Statistical Test
```

Three important tests for students to know are:

- **Z-Test**
- **T-Test**
- **Chi-Square Test**

---

# 2. Z-Test

A **Z-test** is commonly used to test a population mean when the population standard deviation is known or when large-sample conditions justify a normal approximation.

For a one-sample mean test:

$$
Z=
\frac{\bar{x}-\mu_0}
{\sigma/\sqrt n}
$$

Where:

- $\bar{x}$ = sample mean
- $\mu_0$ = hypothesized population mean
- $\sigma$ = population standard deviation
- $n$ = sample size

### Example

Suppose:

$$
\mu_0=50
$$

Sample mean:

$$
\bar{x}=53
$$

Population standard deviation:

$$
\sigma=10
$$

Sample size:

$$
n=100
$$

Then:

$$
Z=
\frac{53-50}
{10/\sqrt{100}}
$$

$$
=\frac{3}{1}
$$

$$
Z=3
$$

The test statistic is:

$$
\boxed{Z=3}
$$

---

# 3. T-Test

A **t-test** is commonly used for comparing means when the population standard deviation is unknown, particularly with smaller samples.

For a one-sample t-test:

$$
t=
\frac{\bar{x}-\mu_0}
{s/\sqrt n}
$$

Where:

- $s$ = sample standard deviation
- $n$ = sample size

### Example

Suppose:

```text id="v3t6fp"
Hypothesized mean = 50
Sample mean       = 53
Sample SD         = 8
Sample size       = 25
```

Then:

$$
t=
\frac{53-50}
{8/\sqrt{25}}
$$

$$
=\frac{3}{1.6}
$$

$$
t=1.875
$$

---

# 4. Common Types of T-Test

### One-Sample T-Test

Compare one sample mean with a known/hypothesized value.

```text id="i9x1l3"
Sample Mean
     ↓
Compare with
     ↓
Known/Hypothesized Mean
```

### Independent Two-Sample T-Test

Compare means of **two independent groups**.

Example:

```text id="j5y3vl"
Group A → Average Marks
Group B → Average Marks
```

Question:

> Is there a significant difference between the two group means?

### Paired T-Test

Used when observations are naturally paired.

Example:

```text id="u8o7io"
Student marks BEFORE training
              ↓
Student marks AFTER training
```

The same students are measured twice.

---

# 5. Chi-Square Test

The **Chi-Square test** is commonly used for **categorical data**.

One major use is testing whether two categorical variables are associated.

### Example

Suppose we want to know whether:

> Gender and product preference are associated.

Data:

| | Product A | Product B |
|---|---:|---:|
| Group 1 | 40 | 20 |
| Group 2 | 25 | 35 |

Both variables are categorical.

We can use a **Chi-Square Test of Independence**.

---

# 6. Chi-Square Test of Independence

It tests:

$$
H_0:
\text{Variables are independent}
$$

against:

$$
H_1:
\text{Variables are associated}
$$

### Example

Variables:

```text id="smw4y7"
Gender
   +
Product Preference
```

The test determines whether the observed relationship is statistically significant.

---

# 7. Chi-Square Formula

The basic statistic is:

$$
\boxed{
\chi^2=
\sum
\frac{(O-E)^2}{E}
}
$$

Where:

- $O$ = observed frequency
- $E$ = expected frequency

The larger the difference between observed and expected frequencies, the larger the Chi-Square statistic tends to be.

---

# 8. Test Selection

A useful high-level guide:

| Situation | Common Test |
|---|---|
| One mean, population SD known | Z-test |
| One mean, population SD unknown | One-sample t-test |
| Two independent means | Independent t-test |
| Before vs after measurements | Paired t-test |
| Two categorical variables | Chi-Square test |
| Categorical association | Chi-Square test |

The exact test should depend on the study design and assumptions.

---

# 9. Python Examples

### T-Test

Using SciPy:

```python
from scipy.stats import ttest_1samp

sample = [52, 48, 51, 55, 49, 53, 50, 54]

stat, p_value = ttest_1samp(sample, popmean=50)

print("t-statistic:", stat)
print("p-value:", p_value)
```

---

### Chi-Square Test

```python
from scipy.stats import chi2_contingency

table = [
    [40, 20],
    [25, 35]
]

chi2, p_value, dof, expected = chi2_contingency(table)

print("Chi-square:", chi2)
print("p-value:", p_value)
```

---

# 10. Data Science Applications

Statistical tests are commonly used for:

### A/B Testing

```text id="z3h2bo"
Version A
    vs
Version B
```

### Feature Analysis

Determine whether a categorical feature is associated with the target.

### Business Analytics

Compare:

- Sales
- Revenue
- Conversion rates
- Customer behavior

### Experimentation

Determine whether an observed change is statistically significant.

---

## Key Points

- **Z-test** → mean testing under appropriate normal/known-SD conditions.
- **T-test** → commonly used for means when population SD is unknown.
- **Independent t-test** → compares two independent means.
- **Paired t-test** → compares paired measurements.
- **Chi-Square test** → commonly used with categorical data.
- Chi-Square statistic:

$$
\chi^2=\sum\frac{(O-E)^2}{E}
$$

- Always choose a statistical test based on the **data type, study design, sample structure, and assumptions**.