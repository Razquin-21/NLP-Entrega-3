import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
BASE_PATH = (
    "C:/Users/d.razquin/Desktop/4ºCURSO/PRIMER CUATRIMESTRE/"
    "Fundamentos del Procesamiento de Lenguaje Natural/PROYECTO/Codigo/"
    "NLP-Entrega-3/datos"
)
OUTPUT_DIR = os.path.join(BASE_PATH, "procesado")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("1. Cargando dataset original y matriz Word2Vec...")
    df = (
        pd.read_parquet(f"{BASE_PATH}/train-00000-of-00003.parquet")
        .drop_duplicates("text")
        .iloc[0:100_000, :]
    )
    df_x = pd.read_csv(f"{BASE_PATH}/word2vec_matrix.csv").set_index("Unnamed: 0")

    print("2. Alineando indices y factorizando etiquetas...")
    df_filtrado = df.loc[df_x.index]
    codes, uniques = pd.factorize(df_filtrado["domain"])
    df_y = pd.Series(codes, index=df_x.index, name="domain")

    print("3. Dividiendo en train y test (85% / 15% estratificado)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df_x.to_numpy(dtype=np.float32),
        df_y.to_numpy(dtype=np.int64),
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=df_y,
    )
    target_names = np.array([str(u) for u in uniques])

    print(f"4. Guardando arrays procesados en: {OUTPUT_DIR}")
    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
    np.save(os.path.join(OUTPUT_DIR, "target_names.npy"), target_names)

    print("Procesamiento completado con exito:")
    print(f"  - X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"  - X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
    print(f"  - Clases ({len(target_names)}): {list(target_names)}")


if __name__ == "__main__":
    main()