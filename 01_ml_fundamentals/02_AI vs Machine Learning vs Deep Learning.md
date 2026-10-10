# Module 1 — Machine Learning Fundamentals

## Topic 2: AI vs Machine Learning vs Deep Learning

### What We Will Cover

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Relationship between AI, ML and DL
- Traditional AI
- Real-life examples
- Key differences
- When to use AI, ML and DL

---

## 1. Artificial Intelligence (AI)

**Artificial Intelligence (AI)** is the broader field of building systems that can perform tasks that normally require human-like intelligence.

These tasks may include:

- Understanding language
- Recognizing images
- Making decisions
- Planning
- Reasoning
- Problem solving
- Generating content

Examples:

- Voice assistants
- Chatbots
- Recommendation systems
- Autonomous systems
- AI-based image recognition

AI is the **broader concept**.

---

# 2. Machine Learning (ML)

**Machine Learning is a subset of Artificial Intelligence.**

Instead of explicitly programming every rule, an ML system learns patterns from data.

```text
Artificial Intelligence
        ↓
Machine Learning
        ↓
Learning from Data
```

### Example

Suppose we want to detect spam emails.

Instead of writing thousands of rules:

```text
IF email contains "free"
AND contains "winner"
AND contains "offer"
THEN spam
```

we can provide historical emails:

| Email | Label |
|---|---|
| Win a free prize | Spam |
| Meeting at 10 AM | Not Spam |
| Claim your free offer | Spam |
| Project report attached | Not Spam |

The ML algorithm learns patterns from these examples.

For a new email:

```text
"You have won a free prize!"
             ↓
          ML Model
             ↓
           Spam
```

---

# 3. Deep Learning (DL)

**Deep Learning is a subset of Machine Learning that uses multi-layer neural networks to learn complex patterns.**

```text
AI
│
└── Machine Learning
     │
     └── Deep Learning
          │
          └── Neural Networks
```

Deep learning is particularly useful for complex data such as:

- Images
- Audio
- Video
- Natural language
- Large-scale unstructured data

### Example: Image Recognition

Suppose we want to identify whether an image contains a cat.

A deep learning model can learn increasingly complex representations:

```text
Image
  ↓
Edges
  ↓
Shapes
  ↓
Patterns
  ↓
Parts of Object
  ↓
Cat
```

The important point is that these representations can be learned automatically from data rather than manually specifying every visual feature.

---

# 4. Relationship Between AI, ML and DL

The easiest way to remember the relationship is:

```text
┌───────────────────────────────────────┐
│       Artificial Intelligence         │
│                                       │
│    ┌─────────────────────────────┐    │
│    │     Machine Learning        │    │
│    │                             │    │
│    │    ┌───────────────────┐    │    │
│    │    │   Deep Learning   │    │    │
│    │    │                   │    │    │
│    │    │ Neural Networks   │    │    │
│    │    └───────────────────┘    │    │
│    └─────────────────────────────┘    │
└───────────────────────────────────────┘
```

Therefore:

> **Every Deep Learning system is Machine Learning, and Machine Learning is a part of AI.**

But:

> **Not every AI system uses Machine Learning, and not every ML system uses Deep Learning.**

---

# 5. Simple Real-Life Comparison

Imagine building a system that plays chess.

### Traditional AI

We could explicitly program strategies and rules:

```text
IF opponent moves here
THEN make this move
```

The intelligence comes largely from **rules designed by humans**.

### Machine Learning

We can provide many examples of chess games.

```text
Chess Games
     ↓
ML Algorithm
     ↓
Learn Patterns
     ↓
Predict Good Moves
```

### Deep Learning

A neural network can learn complex patterns from a very large collection of game positions and outcomes.

```text
Millions of Positions
          ↓
    Neural Network
          ↓
    Learned Patterns
          ↓
      Good Move
```

---

# 6. AI vs ML vs DL

| Feature | AI | ML | DL |
|---|---|---|---|
| Meaning | Artificial Intelligence | Machine Learning | Deep Learning |
| Scope | Broadest | Subset of AI | Subset of ML |
| Main idea | Intelligent behavior | Learn from data | Learn complex representations |
| Rules | May use rules | Learns from data | Learns from large data |
| Feature Engineering | May be manual | Often important | Often reduced/manual features can be learned |
| Data Requirement | Varies | Usually moderate to large | Often large |
| Computation | Varies | Usually moderate | Often high |
| Typical Models | Rule systems, search, ML, etc. | Linear Regression, Decision Tree, Random Forest, SVM | Neural Networks, CNNs, Transformers |
| Common Applications | Planning, reasoning, intelligent agents | Prediction, classification, forecasting | Vision, speech, NLP, generative AI |

---

# 7. Traditional Programming vs ML vs DL

A useful comparison for students:

### Traditional Programming

```text
Rules + Data
     ↓
  Program
     ↓
  Output
```

### Machine Learning

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

### Deep Learning

```text
Large Dataset
      ↓
Neural Network
      ↓
Multiple Layers
      ↓
Learned Representations
      ↓
Prediction
```

---

# 8. When Do We Use ML Instead of Traditional Programming?

Traditional programming is often better when the rules are:

- Clear
- Stable
- Explicit
- Easy to program

Example:

```python
if age >= 18:
    status = "Adult"
else:
    status = "Minor"
```

There is little reason to train an ML model for this.

ML becomes useful when the rules are difficult to explicitly define but patterns can be learned from data.

Examples:

- Spam detection
- Fraud detection
- Customer churn
- House price prediction
- Image classification
- Recommendation systems

---

# 9. When Do We Use Deep Learning?

Deep Learning is particularly useful when:

- The data is complex.
- There is a large amount of training data.
- The problem involves images, audio, video or natural language.
- Neural networks can learn useful representations from the raw or minimally processed data.

Examples:

| Problem | Common Approach |
|---|---|
| House price prediction | Traditional ML |
| Customer churn | Traditional ML |
| Credit risk | Traditional ML |
| Image recognition | Deep Learning |
| Speech recognition | Deep Learning |
| Large language models | Deep Learning |
| Object detection | Deep Learning |

This is a **general guideline**, not a strict rule. Modern systems can combine traditional ML, deep learning, rules, retrieval, search, and other techniques.

---

# 10. The Most Important Relationship

Students should remember:

```text
AI
│
├── Rule-Based Systems
│
├── Search & Planning
│
├── Machine Learning
│     │
│     ├── Regression
│     ├── Classification
│     ├── Clustering
│     └── Ensemble Learning
│
└── Deep Learning
      │
      ├── Neural Networks
      ├── CNNs
      ├── RNNs
      ├── Transformers
      └── Generative Models
```

### Key Points

- **AI** is the broad field of creating intelligent systems.
- **ML** is a subset of AI focused on learning from data.
- **DL** is a subset of ML based on deep neural networks.
- ML is not the same as AI.
- DL is not the same as ML.
- Traditional ML is often effective for structured/tabular business data.
- Deep Learning is especially powerful for complex unstructured data.
- Choosing ML or DL depends on the **problem, data, resources, latency, interpretability requirements, and business constraints**.
