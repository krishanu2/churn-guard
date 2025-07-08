# 📊 ChurnGuard - Telco Customer Churn Prediction App

ChurnGuard is a machine learning-powered Streamlit web app that helps telecom companies predict whether a customer is likely to churn. By analyzing customer attributes and behaviors, the app delivers instant predictions and generates downloadable PDF reports.

---

## 🚀 Features

- ✅ Easy-to-use web interface built with Streamlit
- 📈 Predicts customer churn using a trained Random Forest model
- 🧠 Preprocessing pipeline with OneHotEncoder, Imputation, and Scaling
- 📝 Automatically generates PDF reports with prediction results
- 📂 Model trained on the popular Telco Customer Churn dataset

---

## 🛠️ Tech Stack

- Python 3.10+
- Streamlit
- Scikit-learn
- Pandas
- FPDF
- Joblib

---

## 📁 Folder Structure

```
churn-guard/
│
├── model/                       # Model-related files
│   ├── churn_model.pkl
│   └── feature_columns.pkl
│
├── streamlit_app.py            # Main Streamlit web app
├── train_model.py              # Model training script
├── Telco-Customer-Churn.csv    # Dataset
├── model.joblib                # Trained model with preprocessing pipeline
├── churn_report.pdf            # Sample generated report
└── README.md                   # This file
```

---

## 🧪 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/krishanu2/churn-guard.git
cd churn-guard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:
```bash
pip install streamlit scikit-learn pandas fpdf joblib
```

### 3. Train the model (optional)
```bash
python train_model.py
```

### 4. Run the app
```bash
python -m streamlit run streamlit_app.py
```

---

## 🖼️ Preview

> Screenshot of the app goes here (you can upload later)

---

## 📌 Dataset

The dataset used is the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) available on Kaggle.

---

## 👨‍💻 Author

**Krishanu Mahapatra**  
🔗 [GitHub Profile](https://github.com/krishanu2)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
