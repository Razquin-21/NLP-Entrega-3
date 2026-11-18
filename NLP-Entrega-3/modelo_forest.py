import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
RANDOM_STATE = 42

df = pd.read_parquet("datos/train").drop_duplicates("text").iloc[0:100_000,:]
df["domain"] = pd.factorize(df["domain"])[0]+1
df_y = df["domain"]

w2v_X = pd.read_csv("datos/model/word2vec_matrix.csv").set_index('Unnamed: 0')
df_y = df_y[df_y.index.isin(w2v_X.index)]

X_train, X_test, y_train, y_test = train_test_split(w2v_X,df_y,test_size=0.15,random_state=RANDOM_STATE)
clf = RandomForestClassifier(random_state=RANDOM_STATE)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(classification_report(y_test,y_pred))



