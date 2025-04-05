# Loan Prediction System using Support Vector Machine (SVM)

## 📝 Short Description
This project builds a **Loan Prediction System** using a **Support Vector Machine (SVM)** classifier. It predicts whether a person's loan will be approved or not based on historical loan application data.

## 📊 Dataset
The dataset used is a typical loan application dataset containing attributes like:
- Gender
- Marital Status
- Education
- Applicant Income
- Loan Amount
- Property Area
- Loan Status (Target Variable)

## 📦 Dependencies
```bash
pip install pandas numpy seaborn scikit-learn
```

## 🚀 Project Workflow

### 1. Importing the Dependencies
```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

### 2. Data Collection and Preprocessing
- Load dataset
- Drop missing values
- Label encode categorical variables
- Replace '3+' in dependents with 4
- Convert categorical columns to numeric
- Split data into features (X) and label (Y)

### 3. Data Visualization
- Visualize relationship between Education, Marital Status, and Loan Status using Seaborn

### 4. Train-Test Split
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)
```

### 5. Model Training
- Using Support Vector Classifier (SVC) with a linear kernel
```python
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```

### 6. Model Evaluation
- Evaluate accuracy on both training and testing data
```python
accuracy_score(X_train_prediction, Y_train)
accuracy_score(X_test_prediction, Y_test)
```

### 7. Prediction System
- Input a new applicant's data
- Predict and print whether the loan will be approved or not

## ✅ Example Prediction Output
```python
if prediction[0] == 0:
    print('The person\'s loan is not approved.')
else:
    print('The person\'s loan is approved.')
```

## 🔮 Future Enhancements
- Use other classification models like Random Forest or XGBoost
- Handle missing values using imputation instead of dropping
- Implement a web interface using Flask or Streamlit

## 📄 License
This project is open-source and available under the **MIT License**.

---
💻 Developed using Python, scikit-learn, pandas, and seaborn

