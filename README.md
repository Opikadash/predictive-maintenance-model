## predictive-maintenance-model
Below is the directory structure, files, and their content.
You can create this manually or simply clone from this structure afterwards.

---

## 🔹 📁 Repository Name:

```
predictive-maintenance/
```

---

## 🔹 📁 Project Structure:

```
predictive-maintenance/
├── data/
│ └─ sensor_data.csv
├── src/
│ ├─ preprocess.py
│ ├─ train.py
│ ├─ evaluate.py
│ ├─ deploy.py
├── app.py
├── requirements.txt
├── Dockerfile
├── .gitignore
├── README.md
├── model/
│ ├─ trained_model.pkl
│ ├─ scaler.pkl

```

---

## 🔹 .gitignore:

```gitignore
__pycache__/
venv/
model/
.ipynb_checkpoints/
```

---

## 🔹 requirements.txt:

```txt
pandas
scikit-learn
joblib
streamlit
Flask
matplotlib
```

---

## 🔹 src/preprocess.py:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(data_file):
    df = pd.read_csv(data_file)
    X = df[["temperature", "pressure", "vibration"]]
    y = df["failure"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                        test_size=0.2, 
                                                        random_state=42)
    return X_train, X_test, y_train, y_test, scaler
```

---

## 🔹 src/train.py:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from src.preprocess import preprocess
import joblib

def train(data_file='data/sensor_data.csv'):
    X_train, X_test, y_train, y_test, scaler = preprocess(data_file)

    model = LogisticRegression()
    param_grid = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs', 'saga']
    }
    grid = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    joblib.dump(best_model, "model/trained_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    print(f"Best Params:{grid.best_params}")
    print(f"Training Accuracy:{grid.best_score_*100:.2f}%")
    return best_model, scaler
```

---

## 🔹 src/deploy.py:

```python
import joblib
import numpy as np

def predict_new(temperature, pressure, vibration):
    scaler = joblib.load("model/scaler.pkl")
    model = joblib.load("model/trained_model.pkl")

    X_new = scaler.transform([[temperature, pressure, vibration]])

    prediction = model.predict(X_new)[0]
    return "Failure" if prediction == 1 else "Normal"
```

---

## 🔹 src/evaluate.py:

```python
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import joblib
from src.preprocess import preprocess

def evaluate(data_file='data/sensor_data.csv'):
    X_train, X_test, y_train, y_test, scaler = preprocess(data_file)
    model = joblib.load("model/trained_model.pkl")

    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)
    auc_score = roc_auc_score(y_test, preds)

    fpr, tpr, _ = roc_curve(y_test, preds)

    plt.plot(fpr, tpr, label=f"AUC = {auc_score}")
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("model/roc_curve.png")

    return cm, report, auc_score
```

---

## 🔹 app.py (Streamlit UI)

```python
import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("model/scaler.pkl")
model = joblib.load("model/trained_model.pkl")

st.title("Predict Equipment Failure")
st.write("Enter sensor values:")

temperature = st.number_input("Temperature", 0.0, 100.0, 70.0)
pressure = st.number_input("Pressure", 0.0, 50.0, 30.0)
vibration = st.number_input("Vibration", 0.0, 10.0, 5.0)

if st.button("Predict"):
    X_new = scaler.transform([[temperature, pressure, vibration]])
    prediction = model.predict(X_new)[0]
    result = "Failure" if prediction == 1 else "Normal"

    st.success(f"The equipment is predicted to be: {result}")
```

---

## 🔹 Dockerfile:

```Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["streamlit", "run", "app.py", "--server.port", "5000"]
```

---

## 🔹 README.md:

````markdown
# 🔹 Predictive Maintenance Model 🔹

This repository contains a **end-to-end pipeline** to predict equipment failure based on sensor signals.

---

## 🔹 Features:

✅ Logistic Regression with Hyperparameter Tuning  
✅ Standard Scaling and Preprocessing  
✅ Model Evaluation (confusion matrix, ROC curve)  
✅ Deployment with Streamlit UI  
✅ Dockerized application  

---

## 🔹 Tech Stack:

- **Python**, **Scikit-learn**, **Joblib**
- **Streamlit**, **Flask**
- **Docker**, **Docker Compose**

---

## 🔹 Installation:

```bash
git clone https://github.com/your-username/predictive-maintenance.git
cd predictive-maintenance
pip install -r requirements.txt
````

---

## 🔹 Run Streamlit:

```bash
streamlit run app.py
```

---

## 🔹 Run Docker:

```bash
docker build -t predictive-maintenance .
docker run -p 5000:5000 predictive-maintenance
```

---



🚀 Feel free to contribute, raise issues, or submit pull requests.


```
