

# Wine Quality Analytics: High-Fidelity Classification Pipeline

This project implements a robust machine learning pipeline to classify wine quality based on chemical properties. Moving beyond simple regression, this repository addresses real-world data challenges including **extreme class imbalance**, **feature scaling sensitivity**, and **multi-model optimization**.

## 🚀 Key Engineering Highlights

* **Target Engineering:** Converted ordinal quality scores into a binary "Premium vs. Standard" classification task to align with business-level decision-making.
* **Imbalance Management:** Utilized **SMOTE (Synthetic Minority Over-sampling Technique)** to synthesize minority class samples, significantly improving model recall for high-quality wines.
* **Robust Preprocessing:** Implemented **StandardScaler** (with considerations for **RobustScaler**) to mitigate the influence of chemical outliers on distance-based algorithms.
* **Performance Metrics:** Prioritized **F1-Score** and **Recall** over Accuracy to ensure the model successfully identifies rare "Good" wines without being misled by majority-class bias.

---

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Libraries:** `Pandas`, `NumPy`, `Scikit-Learn`, `Imbalanced-Learn`, `Seaborn`, `Matplotlib`
* **Environment:** Jupyter Notebook / Google Colab

---

## 📊 Pipeline Architecture

1. **Exploratory Data Analysis (EDA):** Visualizing feature correlations and class distributions.
2. **Data Cleaning:** Managing outliers and validating data integrity.
3. **Feature Engineering:** Separating target variables and implementing binary labeling.
4. **Resampling:** Applying SMOTE to the training set to balance the "Premium" class.
5. **Model Benchmarking:** Comparative analysis of 5 architectures:
* Logistic Regression (Baseline)
* K-Nearest Neighbors (KNN)
* Decision Trees
* **Random Forest (Top Performer)**
* Support Vector Machines (SVM)


6. **Hyperparameter Tuning:** Utilizing `GridSearchCV` to optimize the  and  parameters for the final model.

---

## 📈 Results Summary

The **Random Forest** model, combined with SMOTE, yielded the most reliable results for the winery's use case:

* **Accuracy:** ~88%
* **F1-Score:** Balanced performance across both classes.

---

## 🧪 How to Run

1. Clone the repo:
```bash
git clone https://github.com/yourusername/wine-quality-ml.git

```

2. Run the notebook:
```bash
jupyter notebook wine_quality_analysis.ipynb

```



---

## 🧠 Lessons Learned

* **Accuracy is a Lie:** In imbalanced datasets, a model can be "accurate" but useless. Tracking the **Confusion Matrix** is non-negotiable.
* **Data Leakage:** Scaling and oversampling must be performed within the cross-validation folds (or strictly on training data) to maintain test set purity.
