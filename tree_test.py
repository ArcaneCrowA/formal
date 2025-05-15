import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

# import UCI adult dataset
adult = fetch_ucirepo(id=2)

# data (as pandas dataframes)
X: pd.DataFrame = adult.data.features
y: pd.DataFrame = adult.data.targets

#  Convert categorical to numerical
X = pd.get_dummies(X[["workclass", "education", "marital-status", "occupation", "relationship", "race", "native-country"]])

# Split into test and train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Initiate a model
model = DecisionTreeClassifier()

# Train the model
model.fit(X_train, y_train)

# Predict the target
y_pred = model.predict(X_test)

# Print results
print(accuracy_score(y_test, y_pred))
