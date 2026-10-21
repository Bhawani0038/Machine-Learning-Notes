# Module 9: Model Improvement

Module 7 covered how to **evaluate** a model. Module 9 asks the next question: *the model performs badly, or only looks good by luck. How do we make it genuinely better?*

## Roadmap (18 topics in 8 lessons)

Small, closely related topics are merged into one lesson so that no concept is repeated. Each concept is explained fully in one place, and later lessons only give a short reminder.

| Lesson | File | Topics covered | Main question |
|---|---|---|---|
| 1 | `01_Underfitting_and_Overfitting.md` | 1. Underfitting, 2. Overfitting | When does a model learn too little, and when does it memorize? |
| 2 | `02_Bias_Variance_and_Tradeoff.md` | 3. Bias, 4. Variance, 5. Bias-Variance Tradeoff | What is the theory behind these two problems? |
| 3 | `03_Cross_Validation.md` | 6. Cross Validation, 7. K-Fold, 8. Stratified K-Fold | How do we measure performance reliably? |
| 4 | `04_Hyperparameter_Tuning.md` | 9. Hyperparameters, 10. Tuning, 11. GridSearchCV, 12. RandomizedSearchCV | How do we find the best settings? |
| 5 | `05_Regularization.md` | 13. Regularization | How do we control overfitting directly? |
| 6 | `06_Feature_Selection.md` | 14. Feature Selection | Which features should we keep? |
| 7 | `07_Handling_Imbalanced_Data.md` | 15. Imbalanced Data, 16. Class Weights, 17. SMOTE | What if one class is very rare? |
| 8 | `08_Data_Leakage_Prevention.md` | 18. Data Leakage Prevention | How do we avoid fooling ourselves with inflated results? |

## Where each concept is taught (no-repetition map)

- Train/validation gap, the idea of a learning curve: **Lesson 1**
- Bias², variance, the total-error formula: **Lesson 2**
- `cross_val_score`, folds, `StratifiedKFold`: **Lesson 3** (later lessons only use them)
- `Pipeline`, `GridSearchCV`: **Lesson 4** (reused in Lessons 7 and 8)
- L1/L2 penalties, `C` and `alpha`: **Lesson 5**
- The accuracy trap, recall/F1 for imbalance: **Lesson 7** (metric definitions were covered in Module 7)
- The full leakage checklist: **Lesson 8**

## Common dataset

Most lessons use the **Breast Cancer Wisconsin dataset** (`sklearn.datasets.load_breast_cancer`): 569 samples, 30 numeric features, target = malignant/benign. The regularization lesson also uses the diabetes dataset (to show Lasso), and the imbalance lesson uses synthetic data because real datasets rarely show such an extreme imbalance.

## Setup

```bash
pip install scikit-learn pandas numpy matplotlib imbalanced-learn
```

## Images

All images are in the `images/` folder. Keep this folder next to the `.md` files (as provided) so that images appear in the VS Code preview.

## Reading order

Lesson 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Each lesson builds on the previous ones.
