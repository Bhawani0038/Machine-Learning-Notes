# Module 1 — Machine Learning Fundamentals  
## Topic 9: Structured vs Unstructured Data

### What We Will Cover

- What is Structured Data?
- What is Unstructured Data?
- Semi-Structured Data
- Examples
- Structured vs Unstructured vs Semi-Structured
- Machine Learning with different data types
- Real-world examples
- Key Points

---

# 1. Structured Data

**Structured data is data organized in a fixed format, usually rows and columns.**

It is easy to store, search, filter, and analyze.

### Example — Customer Data

| customer_id | name | age | city | salary |
|---:|---|---:|---|---:|
| 101 | Rahul | 28 | Jaipur | 45000 |
| 102 | Priya | 32 | Delhi | 62000 |
| 103 | Amit | 25 | Mumbai | 38000 |

Here:

- Rows represent customers.
- Columns represent attributes.
- Each column has a defined meaning/data type.

This is **structured data**.

### Common Examples

- Excel files
- CSV files
- SQL tables
- Banking transactions
- Sales records
- Employee records
- Customer databases
- Hospital patient tables

---

# 2. Unstructured Data

**Unstructured data does not follow a fixed table-like structure.**

Examples:

- Images
- Videos
- Audio
- Text documents
- Social media posts
- Emails
- PDFs
- CCTV footage

For example, a photograph contains thousands or millions of pixels, but it doesn't naturally have columns like:

```text
age | salary | city | department
```

---

# 3. Text as Unstructured Data

Consider a customer review:

```text
"The delivery was very late and the product
quality was also poor."
```

This is not naturally organized into rows and columns.

However, machine learning can process the text to perform tasks such as:

- Sentiment analysis
- Spam detection
- Customer complaint classification
- Text classification
- Topic detection

---

# 4. Images as Unstructured Data

Consider a medical X-ray.

```text
X-Ray Image
     ↓
ML / Deep Learning Model
     ↓
Normal / Abnormal
```

The image itself is unstructured from a traditional tabular-data perspective.

Computer vision models can learn patterns from the pixels.

---

# 5. Audio as Unstructured Data

Example:

```text
Customer Call Recording
          ↓
Speech Recognition
          ↓
Text
          ↓
Sentiment / Complaint Classification
```

Applications include:

- Voice assistants
- Call-center analysis
- Speech-to-text
- Speaker identification
- Emotion/sentiment analysis

---

# 6. Semi-Structured Data

**Semi-structured data does not follow a strict table format but contains some organization or metadata.**

Common examples:

- JSON
- XML
- HTML
- Log files

### Example JSON

```json
{
  "customer_id": 101,
  "name": "Rahul",
  "city": "Jaipur",
  "orders": [
    {
      "product": "Laptop",
      "amount": 65000
    }
  ]
}
```

This is not a traditional table, but the data has clearly defined keys and structure.

Therefore, JSON is generally considered **semi-structured data**.

---

# 7. Comparison

| Feature | Structured | Semi-Structured | Unstructured |
|---|---|---|---|
| Format | Fixed | Flexible | No fixed structure |
| Common form | Rows/columns | Keys/tags | Free-form |
| Examples | SQL, Excel, CSV | JSON, XML | Images, audio, video, text |
| Easy to query | Yes | Usually | More difficult |
| Schema | Fixed | Flexible | No predefined schema |
| Typical ML data | Tabular | Mixed/nested | Text, image, audio, video |

---

# 8. Real-World Example — E-Commerce

An e-commerce company can have all three types.

### Structured

```text
customer_id
product_id
price
quantity
date
```

Used for:

- Sales analysis
- Customer segmentation
- Sales prediction
- Recommendation systems

### Semi-Structured

```json
{
  "customer_id": 101,
  "device": "mobile",
  "location": {
    "city": "Jaipur"
  }
}
```

Used for:

- Web/application events
- API data
- User activity logs

### Unstructured

```text
Product image
Customer review
Product video
Customer support call
```

Used for:

- Image classification
- Sentiment analysis
- Recommendation
- Speech analysis

---

# 9. Structured Data and Traditional ML

Many traditional machine learning projects work heavily with **structured/tabular data**.

Example:

```text
sqft_area
rooms
parking
location
age
       ↓
ML Model
       ↓
house_price
```

Algorithms such as:

- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting

are commonly used with tabular data.

---

# 10. Unstructured Data and Deep Learning

Deep learning is particularly important when working with complex unstructured data.

### Image

```text
Image
  ↓
Neural Network
  ↓
Object/Class
```

### Text

```text
Text
 ↓
Neural Network
 ↓
Sentiment / Category / Answer
```

### Audio

```text
Audio
  ↓
Neural Network
  ↓
Speech / Classification
```

This does **not** mean traditional ML cannot work with unstructured data. Usually, the raw data needs to be transformed into useful numerical features first.

---

# 11. Important Data Science Point

In machine learning, **models generally work with numerical representations**.

For example:

```text
Customer Review
      ↓
Text Processing
      ↓
Numerical Representation
      ↓
ML Model
      ↓
Prediction
```

Similarly:

```text
Image
  ↓
Pixels / Numerical Representation
  ↓
Model
  ↓
Prediction
```

So the important idea is:

> **Raw data → Representation/Features → ML Model → Prediction**

---

# 12. Quick Identification

### Structured

```text
Excel
CSV
SQL Table
```

### Semi-Structured

```text
JSON
XML
HTML
Logs
```

### Unstructured

```text
Image
Video
Audio
Free-form Text
```

---

## Key Points

- **Structured data → rows and columns.**
- **Semi-structured data → some organization, but flexible format.**
- **Unstructured data → no fixed tabular structure.**
- Excel, CSV, and SQL tables are common structured data.
- JSON and XML are common semi-structured data.
- Images, videos, audio, and free-form text are common unstructured data.
- Traditional ML is widely used for structured/tabular data.
- Deep learning is especially powerful for complex unstructured data.
- ML ultimately needs data represented in a form the model can process.