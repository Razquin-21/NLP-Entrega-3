import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.layers import (Conv1D,Dense,Dropout,Embedding,GlobalMaxPooling1D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from sklearn.preprocessing import LabelEncoder,StandardScaler
RANDOM_STATE = 42

df = pd.read_parquet("datos/train").drop_duplicates("text").iloc[0:100_000,:]

df["domain"] = pd.factorize(df["domain"])[0]+1
df_y = df["domain"]

w2v_X = pd.read_csv("datos/model/word2vec_matrix.csv").set_index('Unnamed: 0')
df_y = df_y[df_y.index.isin(w2v_X.index)]
w2v_X = w2v_X.loc[df_y.index]

le = LabelEncoder()
df_y = le.fit_transform(df_y)
num_classes = len(le.classes_)

X_train, X_test, y_train, y_test = train_test_split(w2v_X.to_numpy(),df_y,test_size=0.15,random_state=RANDOM_STATE)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

embedding_dim = X_train.shape[1]
X_train.reshape((X_train.shape[0],embedding_dim,1))
X_test.reshape((X_test.shape[0],embedding_dim,1))


model = Sequential([
    Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation="softmax"),
])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(X_train,y_train,epochs=15,batch_size=128,validation_split=0.1)
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred,axis=1)
print(classification_report(y_test,y_pred,target_names=[str(c) for c in le.classes_]))

