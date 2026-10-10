# Module 2 — Mathematics & Statistics for Machine Learning
## Topic 1: Population vs Sample

### What We Will Cover

- Population
- Sample
- Parameter
- Statistic
- Census
- Sampling
- Why samples are used
- Real-life ML/Data Science examples
- Population vs Sample comparison
- Key Points

---

# 1. What is a Population?

A **population** is the complete group of individuals, objects, or observations that we are interested in studying.

### Example — Students

Suppose a college has **5,000 students**.

If we want to study the average height of **all 5,000 students**:

```text
Population = All 5,000 students
```

---

# 2. What is a Sample?

A **sample** is a smaller group selected from the population.

Suppose we select 200 students from those 5,000 students:

```text
Population = 5,000 students
Sample     = 200 students
```

We use the sample to learn something about the larger population.

---

# 3. Simple Example

Suppose an e-commerce company has:

```text
1,000,000 customers
```

It wants to know:

> What percentage of customers are satisfied with our service?

Instead of asking all 1,000,000 customers, the company may survey:

```text
5,000 customers
```

Therefore:

```text
1,000,000 → Population
5,000     → Sample
```

---

# 4. Why Do We Use Samples?

Studying the entire population can be:

- Expensive
- Time-consuming
- Difficult
- Sometimes impossible

### Example

To determine the quality of every product manufactured by a factory, testing every single product may be impractical.

Instead:

```text
Factory Production
       ↓
Select Sample
       ↓
Test Sample
       ↓
Estimate Overall Quality
```

---

# 5. Census

When we collect information from **every member of the population**, it is called a **census**.

Example:

```text
Population = 10,000 employees

Information collected from:
10,000 employees
       ↓
Census
```

If information is collected from only 500 employees:

```text
500 employees
     ↓
Sample
```

---

# 6. Parameter

A **parameter** is a numerical value that describes the **population**.

For example, if the average salary of all 10,000 employees is:

```text
₹55,000
```

then ₹55,000 is a **population parameter**.

Common population parameters:

- Population mean → `μ`
- Population variance → `σ²`
- Population standard deviation → `σ`

---

# 7. Statistic

A **statistic** is a numerical value calculated from a **sample**.

Suppose we select 500 employees and calculate their average salary:

```text
Sample mean = ₹52,000
```

₹52,000 is a **sample statistic**.

Common sample statistics:

- Sample mean → `x̄`
- Sample variance → `s²`
- Sample standard deviation → `s`

---

# 8. Parameter vs Statistic

| Parameter | Statistic |
|---|---|
| Describes population | Describes sample |
| Usually unknown | Calculated from sample |
| Population mean → `μ` | Sample mean → `x̄` |
| Population SD → `σ` | Sample SD → `s` |
| Based on entire population | Based on selected sample |

### Easy Memory Trick

```text
Population → Parameter
Sample     → Statistic
```

---

# 9. Population → Sample → Conclusion

This is a very important statistical idea:

```text
Population
     ↓
Select Sample
     ↓
Collect Data
     ↓
Calculate Statistics
     ↓
Make Inference
     ↓
Population
```

For example:

```text
100,000 customers
       ↓
2,000 customers
       ↓
Calculate average satisfaction
       ↓
Estimate overall customer satisfaction
```

This process is called **statistical inference**.

---

# 10. Sampling in Machine Learning

Machine learning datasets are often treated as samples from some larger real-world population.

Example:

Suppose we want to build a model that predicts house prices.

Our dataset contains:

```text
10,000 houses
```

But the real-world population could contain:

```text
Millions of houses
```

Our dataset is therefore a sample of the broader population we want the model to work on.

```text
Real-world Houses
      ↓
   Dataset
      ↓
Training / Testing
      ↓
ML Model
      ↓
New Houses
```

The model should ideally **generalize** beyond the specific houses it saw during training.

---

# 11. Example — Medical Research

Suppose there are:

```text
Population = 1,000,000 people
```

Researchers select:

```text
Sample = 10,000 people
```

They calculate:

```text
Average blood pressure
Disease rate
Age distribution
```

They use these sample statistics to understand the larger population.

---

# 12. Population vs Sample

| Population | Sample |
|---|---|
| Entire group | Part of the group |
| Usually larger | Usually smaller |
| Parameter describes it | Statistic describes it |
| Can be expensive to study | Usually cheaper |
| Census studies population | Sampling studies a subset |

---

## Key Points

- **Population = complete group of interest.**
- **Sample = smaller group selected from the population.**
- **Census = studying the entire population.**
- **Parameter = numerical measure of a population.**
- **Statistic = numerical measure of a sample.**
- We use samples because studying the entire population is often impractical.
- In ML, a dataset can be viewed as a sample of the larger real-world data population.
- Good sampling helps models and statistical conclusions **generalize** to unseen data.