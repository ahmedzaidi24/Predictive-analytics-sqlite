# Predictive Analytics on SQLite Dataset 

## Overview
This repository contains a comprehensive data mining project developed as part of the **Data Mining** unit at Curtin University. The project involves building predictive classification models using real-world machine learning workflows on a completely anonymized and obfuscated SQLite dataset.

The aim was to predict unknown class labels from the test data using various data preprocessing, modeling, and evaluation techniques — aligning directly with the real-world tasks of a data analyst/scientist.

---

## Objectives
- Clean and preprocess a complex SQLite dataset containing missing values, duplicates, and mixed data types
- Handle data imbalance using **SMOTE** (Synthetic Minority Oversampling)
- Apply dimensionality reduction and transformation techniques (e.g., binarization, standardization)
- Train and tune three classification models: **K-Nearest Neighbors**, **Decision Trees**, and **Naive Bayes**
- Evaluate model performance via **Stratified K-Fold**, **K-Fold**, and **ShuffleSplit** cross-validation
- Export final predictions to a new SQLite file for assessment

---

## Data Source
- Format: SQLite (`Assignment2023.sqlite`)
- Tables: `train` (5000 samples), `test` (500 samples)
- Class column (`0`, `1`, `2`) present in train, but missing in test
- All attribute names are obfuscated (e.g., ATT01, ATT02...)

---

## Workflow
### 📊 1. Data Preparation
- **Missing Values:** Identified and treated based on missing percentage; columns over 50% dropped, others interpolated
- **Duplicates:** Checked for row-wise and column-wise duplications
- **Categorical Attributes:** Binarized using `pandas.get_dummies()`
- **Standardization:** Z-score normalization using `StandardScaler`

### 🧪 2. Model Training
- **KNN:** Tuned `n_neighbors` and `weights` using GridSearchCV
- **Decision Tree:** Tuned `criterion` and `min_samples_split`
- **Naive Bayes:** Baseline probabilistic classifier
- **Validation:** Used 3 techniques for fairness:
  - `StratifiedKFold`
  - `KFold`
  - `ShuffleSplit`

### ⚖️ 3. Data Balancing
- Applied **SMOTE** to equalize class distribution before model training

### 🧮 4. Evaluation
| Model              | Accuracy (Raw) | Accuracy (SMOTE) |
|-------------------|----------------|------------------|
| KNN               | 86.48%         | 81.92%           |
| Decision Tree     | 73.56%         | 71.36%           |
| Naive Bayes       | 73.2%          | -                |

- Final predictions were exported to `Answers.sqlite` with columns: `Index`, `Predict1`, `Predict2`

---

## Repository Contents
| File | Description |
|------|-------------|
| `notebook_Zaidi_20972008.ipynb` | Complete predictive workflow in Jupyter Notebook |
| `Answers_20972008.sqlite` | Final prediction output file as per assignment format |
| `Report_Zaidi_20972008.pdf` | Final report detailing methodology, analysis, and results |
| `Requirements.pdf` | Assignment specification and rubric |

---

## Technologies Used
- `Python`
- `pandas`, `numpy`, `scikit-learn`, `imblearn`
- `matplotlib`, `seaborn`
- `sqlite3`, `sqlalchemy`

---

## Author
**Syed Muhammad Ahmed Zaidi**  


---

## License
This project is part of academic coursework and is intended for educational and illustrative purposes only.
