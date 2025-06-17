# src/evaluate.py

from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import joblib
from src.preprocessing import preprocess

def evaluate(data_file='data/sensor_data.csv'):
    X_train, X_test, y_train, y_test, scaler = preprocess(data_file)
    model = joblib.load("model/trained_model.pkl")

    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)
    auc_score = roc_auc_score(y_test, preds)

    fpr, tpr, _ = roc_curve(y_test, preds)

    # Plot ROC curve
    plt.plot(fpr, tpr, label=f"AUC = {auc_score}")
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig("model/roc_curve.png")

    return cm, report, auc_score
