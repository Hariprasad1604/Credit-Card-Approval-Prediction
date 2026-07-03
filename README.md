# Credit Card Approval Prediction

A machine learning web application that predicts whether a credit card application is likely to be approved based on applicant information.

The project includes data preprocessing, exploratory data analysis (EDA), feature engineering, model comparison, and deployment using a Flask web application.



## 🌐 Live Demo

**https://credit-card-approval-prediction-ltmd.onrender.com**

## 📂 GitHub Repository

https://github.com/Hariprasad1604/Credit-Card-Approval-Prediction

---

## Features

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Handling Missing Values
- Outlier Analysis
- Label Encoding
- One-Hot Encoding
- Feature Scaling
- SMOTE for Class Imbalance
- Model Comparison
- Best Model Selection
- Flask Web Application
- Responsive User Interface

---

## Dataset

- application_record.csv
- credit_record.csv

---

## Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

The best-performing model is saved and used for prediction.

---

## Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Flask
- Joblib

### Frontend

- HTML
- CSS

---

## Project Structure

```
Credit Card Approval Prediction
│
├── app.py
├── task.ipynb
├── requirements.txt
├── README.md
│
├── application_record.csv
├── credit_record.csv
│
├── model
│   ├── binary_encoders.pkl
│   ├── credit_card_approval_model.pkl
│   ├── feature_columns.pkl
│   └── scaler.pkl
│
├── templates
│   └── index.html
│
└── static
    └── style.css
```

---

## Installation

Clone the repository

```bash
git clone <repository-url>
```

Move into the project folder

```bash
cd Credit-Card-Approval-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the Flask server

```bash
python app.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

## Screenshots

Add screenshots of:

- Home Page
- Prediction Result
- Approval
- Rejection

---

## Future Improvements

- Deploy on Render
- Deploy REST API
- Improve UI
- Add User Authentication
- Model Monitoring

---

## Author

**Hari Prasad Tankala**

GitHub: https://github.com/your-github-Hariprasad1604