# Module 2 — Mathematics & Statistics for ML

## Topic 18: Central Limit Theorem (CLT)

### What We Will Cover
- Central Limit Theorem
- Population vs Sample Mean
- Sampling Distribution
- Standard Error
- CLT Intuition
- Why CLT is Important in ML
- Practical Example
- Python Simulation

---

## 1. What is Central Limit Theorem?

The **Central Limit Theorem (CLT)** says:

> When we take many random samples from a population and calculate the mean of each sample, the distribution of those sample means tends to become approximately **normal**, especially as the sample size increases.

Even if the original population is **not normally distributed**, the distribution of sample means can become approximately normal.

---

## 2. Simple Intuition

Suppose a company's employee salaries are highly skewed:

```text
₹20K, ₹22K, ₹25K, ₹27K, ₹30K, ₹35K, ₹40K, ₹200K
```

The population itself is **not normally distributed**.

Now repeatedly take random samples of 30 employees:

```text
Sample 1 → Mean = ₹42K
Sample 2 → Mean = ₹45K
Sample 3 → Mean = ₹41K
Sample 4 → Mean = ₹44K
Sample 5 → Mean = ₹43K
...
```

If we collect thousands of these sample means, their distribution tends to look approximately **bell-shaped/normal**.

---

## 3. Population vs Sampling Distribution

### Population Distribution

The actual distribution of individual values:

```text
Population
   ↓
Individual salaries
   ↓
May be skewed / non-normal
```

### Sampling Distribution

Distribution of means obtained from many samples:

```text
Population
   ↓
Take many samples
   ↓
Calculate mean of each sample
   ↓
Distribution of sample means
   ↓
Approximately Normal
```

### Important

**CLT does NOT say that the original data becomes normal.**

It says that the **distribution of sample means** becomes approximately normal.

---

## 4. Standard Error

The standard deviation of the sampling distribution of the mean is called the **Standard Error (SE)**.

For population standard deviation:

$$
SE = \frac{\sigma}{\sqrt{n}}
$$

Where:

- $\sigma$ = population standard deviation
- $n$ = sample size
- $SE$ = standard error

### Example

Suppose:

```text
σ = 20
n = 100
```

Then:

$$
SE = \frac{20}{\sqrt{100}}
$$

$$
SE = \frac{20}{10}=2
$$

So the standard error is **2**.

---

## 5. Effect of Sample Size

As sample size increases:

$$
n \uparrow
$$

then:

$$
SE \downarrow
$$

Example:

| Sample Size | Standard Error if σ=20 |
|---:|---:|
| 25 | 4 |
| 100 | 2 |
| 400 | 1 |
| 900 | 0.67 |

Larger samples generally produce **more stable sample means**.

---

## 6. Why CLT is Important

CLT is important because it allows us to make statistical conclusions about a population using samples.

It is a foundation for:

- Confidence intervals
- Hypothesis testing
- Estimation
- Statistical inference
- A/B testing
- Business experiments
- Model evaluation and comparison

---

## 7. Python Intuition

```python
import numpy as np
import matplotlib.pyplot as plt

population = np.random.exponential(scale=10, size=100000)

sample_means = []

for _ in range(1000):
    sample = np.random.choice(population, size=30)
    sample_means.append(sample.mean())

plt.hist(sample_means, bins=30)
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")
plt.title("Sampling Distribution of Sample Means")
plt.show()
```

The original population is **skewed**, but the distribution of the **sample means** tends to look much more like a normal distribution.

---

## 8. CLT in One Diagram

```text
        Population
       /           \
   Non-Normal     Skewed
        |
        | Take many random samples
        ↓
   Sample 1 → Mean
   Sample 2 → Mean
   Sample 3 → Mean
   Sample 4 → Mean
       ...
        ↓
 Distribution of Sample Means
        ↓
 Approximately Normal
```

---

## 9. Important Interview Point

### ❌ Incorrect

> CLT says every dataset becomes normally distributed when sample size increases.

### ✅ Correct

> CLT says that the **sampling distribution of the sample mean** tends toward a normal distribution as sample size increases, under appropriate conditions.

---

## Key Points

- CLT deals with **sample means**.
- Original population does **not** have to be normal.
- Larger sample size → sampling distribution becomes more stable.
- Standard Error:

$$
SE=\frac{\sigma}{\sqrt n}
$$

- CLT is a foundation of **confidence intervals and hypothesis testing**.
- **Population distribution ≠ sampling distribution**.
- CLT is extremely important for statistical inference in Data Science and ML.