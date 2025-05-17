from time import time

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df: pd.DataFrame = pd.read_csv("adult.csv")

df = df.dropna()
X = df.drop("income", axis=1)
y = df["income"]
y = y.map(lambda x: {"<=50K": 1, ">50K": 0}.get(x))

categorical = ["workclass", "education", "marital.status", "occupation", "relationship", "race", "sex", "native.country"]
for col in categorical:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Split into test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Initiate a model
model = DecisionTreeClassifier()

m_acc: list[float] = []
m_prec: list[float] = []
m_time: list[float] = []
start: float = time()

# Train the model and measure time and accuracy and precision
for _ in range(100):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    m_time.append(time() - start)
    m_acc.append(accuracy_score(y_test, y_pred))
    m_prec.append(precision_score(y_test, y_pred))

# Print results
print("Mean accuracy:", sum(m_acc) / 100)
print("Mean Precision:", sum(m_prec) / 100)
print("\nMean spent time in seconds:", sum(m_time) / 100)
