# Module 2 — Mathematics & Statistics for ML

## Topic 19: Hypothesis Testing

### What We Will Cover
- What is Hypothesis Testing?
- Null Hypothesis
- Alternative Hypothesis
- Test Statistic
- p-value
- Significance Level
- Decision Making
- Type I & Type II Errors
- ML / Data Science Examples

---

## 1. What is Hypothesis Testing?

**Hypothesis testing** is a statistical method used to determine whether there is enough evidence in sample data to support a claim about a population.

### Simple Example

A company claims:

> "Our new teaching method improves student marks."

We collect marks from students and test whether the improvement is statistically significant.

---

## 2. Two Hypotheses

Every hypothesis test generally starts with two competing statements.

### Null Hypothesis — H₀

The **null hypothesis** represents the default assumption.

Example:

> H₀: The new teaching method does **not** improve marks.

### Alternative Hypothesis — H₁ / Hₐ

The alternative hypothesis represents what we want to investigate.

> H₁: The new teaching method **does** improve marks.

```text
H₀ → No significant effect
H₁ → Significant effect
```

---

## 3. Example: Average Student Marks

Suppose the historical average marks are:

$$
\mu = 60
$$

After introducing a new teaching method, we take a sample of students.

Sample average:

$$
\bar{x}=65
$$

We want to know:

> Is 65 significantly different from 60, or could this difference happen just because of random sampling?

This is where hypothesis testing is useful.

---

## 4. Basic Hypothesis Testing Process

```text
Define Problem
      ↓
Set H₀ and H₁
      ↓
Choose Significance Level α
      ↓
Collect Sample Data
      ↓
Calculate Test Statistic
      ↓
Calculate p-value
      ↓
Compare p-value with α
      ↓
Make Decision
```

---

## 5. Significance Level — α

The **significance level** is the threshold we use for deciding whether the evidence is strong enough to reject H₀.

Common value:

$$
\alpha = 0.05
$$

This means we accept a **5% significance level**.

Other common values:

```text
α = 0.10
α = 0.05
α = 0.01
```

---

## 6. What is a p-value?

The **p-value** measures how compatible the observed data is with the null hypothesis.

Simple classroom intuition:

> A small p-value means the observed result would be relatively unusual if H₀ were true.

### Decision Rule

| p-value | Decision |
|---:|---|
| p ≤ 0.05 | Reject H₀ |
| p > 0.05 | Fail to reject H₀ |

### Important

We usually say:

**"Fail to reject H₀"**

rather than:

**"Accept H₀"**

because a test does not prove that H₀ is absolutely true.

---

## 7. Simple Example

Suppose:

```text
H₀: New method does not improve marks
H₁: New method improves marks

α = 0.05
p-value = 0.02
```

Since:

$$
0.02 < 0.05
$$

we **reject H₀**.

Conclusion:

> There is statistically significant evidence that the new method improves marks.

---

## 8. Type I and Type II Errors

Hypothesis testing can make two important types of errors.

| Reality | Our Decision | Result |
|---|---|---|
| H₀ is true | Reject H₀ | **Type I Error** |
| H₀ is true | Fail to reject H₀ | Correct |
| H₀ is false | Reject H₀ | Correct |
| H₀ is false | Fail to reject H₀ | **Type II Error** |

### Type I Error

Rejecting H₀ when H₀ is actually true.

Also called:

> **False Positive**

### Type II Error

Failing to reject H₀ when H₀ is actually false.

Also called:

> **False Negative**

---

## 9. Real-Life Example

Medical test:

```text
H₀: Patient does not have disease
H₁: Patient has disease
```

### Type I Error

Test says disease exists, but patient is actually healthy.

```text
False Positive
```

### Type II Error

Test says patient is healthy, but patient actually has the disease.

```text
False Negative
```

---

## 10. Hypothesis Testing in Data Science

Hypothesis testing is commonly used in:

- A/B Testing
- Marketing experiments
- Product experiments
- Business analysis
- Medical studies
- Feature analysis
- Model comparison
- Statistical inference

### Example: A/B Testing

```text
Version A → Conversion rate = 8%
Version B → Conversion rate = 10%
```

Question:

> Is B genuinely better, or could the difference be due to random variation?

Hypothesis testing helps answer this.

---

## Key Points

- Hypothesis testing helps make decisions using sample data.
- **H₀** = default/no-effect assumption.
- **H₁** = alternative claim.
- **α** = significance level, commonly 0.05.
- **p-value** measures evidence against H₀.
- If **p ≤ α → Reject H₀**.
- If **p > α → Fail to reject H₀**.
- Type I Error = **False Positive**.
- Type II Error = **False Negative**.
- Hypothesis testing is heavily used in **Data Science, A/B testing, research, and statistical inference**.