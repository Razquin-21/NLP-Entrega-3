import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization

RANDOM_STATE = 42

# 1. Carga y limpieza de datos
df = pd.read_parquet("datos/train").drop_duplicates("text").iloc[0:100_000, :]
df["domain"] = pd.factorize(df["domain"])[0] + 1
df_y = df["domain"]

w2v_X = pd.read_csv("datos/model/word2vec_matrix.csv").set_index('Unnamed: 0')
df_y = df_y[df_y.index.isin(w2v_X.index)]
w2v_X = w2v_X.loc[df_y.index]

le = LabelEncoder()
df_y = le.fit_transform(df_y)
num_classes = len(le.classes_)
print(f"Número de clases: {num_classes}")

# 2. División y escalado
X_train, X_test, y_train, y_test = train_test_split(
    w2v_X.to_numpy(), df_y, test_size=0.15, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Reshape obligatorio a 3D: (muestras, longitud_vector, canales)
embedding_dim = X_train.shape[1]
X_train = X_train.reshape((X_train.shape[0], embedding_dim, 1))
X_test = X_test.reshape((X_test.shape[0], embedding_dim, 1))

# 4. Arquitectura CNN 1D
model = Sequential([
    # Primera capa convolucional
    Conv1D(filters=128, kernel_size=3, activation="relu", padding="same", input_shape=(embedding_dim, 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # Segunda capa convolucional para extraer patrones más complejos
    Conv1D(filters=64, kernel_size=3, activation="relu", padding="same"),
    BatchNormalization(),
    GlobalMaxPooling1D(),
    Dropout(0.4),

    # Capa densa de clasificación
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

# 5. Compilación y entrenamiento
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(X_train, y_train,epochs=15,batch_size=128,validation_split=0.1)

# 6. Evaluación
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))
