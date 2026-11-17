import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

RANDOM_STATE = 42
BASE_PATH = (
    "C:/Users/d.razquin/Desktop/4ºCURSO/PRIMER CUATRIMESTRE/"
    "Fundamentos del Procesamiento de Lenguaje Natural/PROYECTO/Codigo/"
    "NLP-Entrega-3/datos"
)
PROCESSED_DIR = os.path.join(BASE_PATH, "procesado")
MODELS_DIR = os.path.join(BASE_PATH, "modelos")


def load_processed_data():
    required = ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy", "target_names.npy"]
    for fname in required:
        if not os.path.exists(os.path.join(PROCESSED_DIR, fname)):
            raise FileNotFoundError(
                f"No se encontró '{fname}' en {PROCESSED_DIR}. "
                "Ejecuta primero 'procesamiento.py'."
            )

    X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))
    target_names = np.load(os.path.join(PROCESSED_DIR, "target_names.npy"), allow_pickle=True)
    return X_train, X_test, y_train, y_test, target_names


def main():
    X_train, X_test, y_train, y_test, target_names = load_processed_data()

    print("Entrenando Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)
    print("=== Resultados Logistic Regression ===")
    print(f"F1 Macro: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(classification_report(y_test, y_pred, target_names=target_names))

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "logistic_regression.joblib")
    joblib.dump(lr_model, model_path)
    print(f"Modelo guardado en: {model_path}")


if __name__ == "__main__":
    main()