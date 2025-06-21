# 📊 Customer Churn Prediction

A machine learning web application to predict whether a customer is likely to churn based on their usage and service patterns.

---

## 🚀 Project Overview

Customer churn is a critical business problem for telecom and subscription-based companies. This project aims to predict whether a customer will churn or stay using machine learning models trained on historical customer data.

The solution includes:
- Data preprocessing and feature engineering
- Model training using Random Forest Classifier
- A deployed Streamlit app for predictions

---

## 🧠 Technologies Used

- **Python 3.10+**
- **Pandas**, **NumPy**, **Scikit-learn**
- **Matplotlib**, **Seaborn**
- **Streamlit** (for frontend web app)
- **Joblib** (for model serialization)
- **Jupyter Notebook** (for training and exploration)

---

## 📁 Project Structure

```
customer_churn_prediction/
│
├── app/
│   └── app.py                   # Streamlit frontend
│
├── data/
│   └── churn_data.csv           # Raw dataset (Telco Churn)
│
├── model/
│   ├── churn_rf_model.pkl       # Trained Random Forest model
│   └── feature_columns.pkl      # Features used in training
│
├── notebook/
│   └── Customer Churn Notebook.ipynb  # Model training notebook
│
├── venv/                        # Virtual environment (optional)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 💠 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/customer_churn_prediction.git
cd customer_churn_prediction
```

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate       # On Windows
# OR
source venv/bin/activate    # On macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Model Training (Optional)

If you want to retrain the model:

```bash
# Launch Jupyter Notebook
jupyter notebook

# Open the notebook:
notebook/Customer Churn Notebook.ipynb

# Run all cells to preprocess data and save new model
```

---

## 💻 Running the Streamlit App

```bash
streamlit run app/app.py
```

> This will launch the web app in your default browser where you can input customer details and predict churn.

---

## 🧪 Example Inputs to Test

Try customers with:
- Low tenure
- High monthly charges
- No backup/security/tech support
- Month-to-month contract  
They are likely to be predicted as **churners**.

---

## 📆 requirements.txt (if needed)

To create one:

```bash
pip freeze > requirements.txt
```

Sample contents:

```text
joblib
matplotlib
numpy
pandas
scikit-learn
seaborn
streamlit
```

---

## 🧑‍💻 Author

**Rohan Suryawanshi**  
Junior Data Engineer | AI/ML Enthusiast

---

