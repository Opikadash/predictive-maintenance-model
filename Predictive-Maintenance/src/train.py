# train.py
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from src.preprocessing import preprocess
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

