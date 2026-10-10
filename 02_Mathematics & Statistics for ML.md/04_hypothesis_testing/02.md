# Module 2 — Mathematics & Statistics for ML

## Topic 20: Confidence Interval

### What We Will Cover
- What is a Confidence Interval?
- Point Estimate
- Interval Estimate
- Confidence Level
- Margin of Error
- 95% Confidence Interval
- Relationship with Standard Error
- Practical Example
- ML / Data Science Use

---

## 1. What is a Confidence Interval?

A **confidence interval (CI)** gives a range of plausible values for an unknown population parameter based on sample data.

Instead of saying:

> Average salary = ₹50,000

we can say:

> Estimated average salary is around **₹50,000**, with a **95% confidence interval** of ₹48,000–₹52,000.

So, a confidence interval gives:

```text
Point Estimate ± Margin of Error
```

---

## 2. Point Estimate vs Interval Estimate

### Point Estimate

A single value used to estimate a population parameter.

Example:

```text
Sample mean = ₹50,000
```

### Interval Estimate

A range of possible values.

```text
₹48,000 ───────── ₹52,000
          ↑
      Sample Mean
```

The interval provides more information than a single number.

---

## 3. Confidence Level

Common confidence levels are:

| Confidence Level | Meaning |
|---:|---|
| 90% | Less confidence, narrower interval |
| 95% | Commonly used |
| 99% | Higher confidence, wider interval |

Generally:

```text
Confidence ↑
      ↓
Interval Width ↑
```

For the same data, a **99% confidence interval** is usually wider than a **95% interval**.

---

## 4. Example

Suppose we survey 100 students.

Sample average:

$$
\bar{x}=70
$$

Standard deviation:

$$
\sigma=10
$$

Sample size:

$$
n=100
$$

Standard Error:

$$
SE=\frac{\sigma}{\sqrt n}
$$

$$
SE=\frac{10}{10}=1
$$

For an approximate 95% interval using the normal critical value:

$$
\bar{x}\pm1.96(SE)
$$

Therefore:

$$
70\pm1.96
$$

So:

$$
68.04 \leq \mu \leq 71.96
$$

The approximate 95% confidence interval is:

**68.04 to 71.96**

---

## 5. Margin of Error

The margin of error tells us how far the interval extends from the point estimate.

For an approximate normal-based interval:

$$
\text{Margin of Error} = z^* \times SE
$$

Therefore:

$$
\text{CI} = \bar{x}\pm z^*SE
$$

For a 95% confidence level:

$$
z^*\approx1.96
$$

---

## 6. Relationship with Sample Size

Remember:

$$
SE=\frac{\sigma}{\sqrt n}
$$

If sample size increases:

$$
n\uparrow
$$

then:

$$
SE\downarrow
$$

Therefore the confidence interval generally becomes **narrower**.

Example:

```text
Small Sample
     ↓
Larger uncertainty
     ↓
Wider CI


Large Sample
     ↓
Smaller uncertainty
     ↓
Narrower CI
```

---

## 7. Very Important Interpretation

A common simplified classroom explanation is:

> "We are 95% confident that the population parameter lies between these two values."

More precisely, in frequentist statistics, the **95% refers to the long-run success rate of the interval-building procedure**: if we repeatedly took samples and constructed intervals using the same method, about 95% of those intervals would contain the true parameter.

The true population parameter is treated as fixed; the interval changes from sample to sample.

---

## 8. Confidence Interval in Data Science

Suppose an e-commerce company tests a new website.

```text
Old website conversion = 8%
New website conversion = 10%
```

Instead of looking only at the observed 2% difference, we can calculate confidence intervals.

This helps answer:

> How uncertain is our estimate?

Confidence intervals are useful for:

- A/B testing
- Business analytics
- Survey analysis
- Model performance estimates
- Experimentation
- Statistical inference

---

## 9. Confidence Interval vs Hypothesis Testing

| Confidence Interval | Hypothesis Testing |
|---|---|
| Gives a range | Tests a claim |
| Shows uncertainty | Gives a decision framework |
| Estimates parameter | Tests H₀ |
| Example: 68–72 | Example: Reject H₀ |

They are closely related.

---

## Key Points

- Confidence interval gives a **range of plausible values** for a population parameter.
- Basic structure:

$$
\text{CI} = \text{Estimate} \pm \text{Margin of Error}
$$

- Standard Error:

$$
SE=\frac{\sigma}{\sqrt n}
$$

- Larger sample → smaller SE → generally narrower CI.
- Higher confidence level → generally wider CI.
- **95% CI** is commonly used.
- Confidence intervals communicate **uncertainty**, which is important in Data Science and ML experiments.