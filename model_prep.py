import pandas as pd
import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("DSP_6.csv")
train.drop(columns=["Cabin"], inplace=True)
train.fillna(train.mean(numeric_only=True), inplace=True)
train.dropna(inplace=True)
sex = pd.get_dummies(train["Sex"], drop_first=True, dtype="int")
lab_enc = LabelEncoder()
train["Embarked"] = lab_enc.fit_transform(train["Embarked"])
train = pd.concat([train, sex], axis=1)
train.drop(["Sex", "Name", "Ticket", "PassengerId"], axis=1, inplace=True)

## model training
X = train.drop(["Survived"], axis=1)
y = train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10,random_state=101)

def model(X_train, y_train):
  forest = RandomForestClassifier(n_estimators=10, random_state=101)
  forest.fit(X_train, y_train)
  print(f"Las: {forest.score(X_train, y_train)}")

  lreg = LogisticRegression(max_iter=500)
  lreg.fit(X_train, y_train)
  print(f"Regresja logistyczna: {lreg.score(X_train, y_train)}")

  tree = DecisionTreeClassifier()
  tree.fit(X_train, y_train)
  print(f"Drzewa decyzyjne: {tree.score(X_train, y_train)}")

  return forest, lreg, tree

forest, lreg, tree = model(X_train, y_train)

y1_predict = forest.predict(X_test)
print(f"Random Forest {accuracy_score(y_test, y1_predict)}")

y2_predict = lreg.predict(X_test)
print(f"Logistic Regresion {accuracy_score(y_test, y2_predict)}")

y3_predict = tree.predict(X_test)
print(f"Desision Tree {accuracy_score(y_test, y3_predict)}")

filename = "model.h5"
pickle.dump(lreg, open(filename, "wb"))