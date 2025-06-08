# 🩺 Diabetes Prediction Using Logistic Regression

A machine learning project to predict diabetes using clinical health data. This project uses logistic regression to classify whether a person is diabetic or non-diabetic based on various medical attributes.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Sample Prediction](#sample-prediction)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## 📖 Overview

This project aims to assist in the early detection of diabetes using a logistic regression model trained on the Pima Indians Diabetes Dataset. It covers the complete machine learning pipeline including data preprocessing, model training, evaluation, and making predictions on new data.

---

## 📂 Dataset

Dataset used: [Pima Indians Diabetes Dataset](https://github.com/YBI-Foundation/Dataset/raw/main/Diabetes.csv)

### Features:

- Pregnancies
- Glucose
- Blood Pressure (Diastolic)
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function (DPF)
- Age
- Diabetes Outcome (0 = Non-diabetic, 1 = Diabetic)

---

## ⚙️ Technologies Used

- Python 3
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook (optional)

---

## ✅ Features

- Data loading, exploration, and cleaning
- Normalization using `MinMaxScaler`
- Train/test split with stratification
- Model building with `LogisticRegression`
- Evaluation using confusion matrix and classification report
- Prediction on new patient sample

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/diabetes-prediction-logistic-regression.git
cd diabetes-prediction-logistic-regression
pip install -r requirements.txt
