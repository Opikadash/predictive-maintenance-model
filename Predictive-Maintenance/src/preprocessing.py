# src/preprocess.py
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
