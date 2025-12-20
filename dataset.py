import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sensitive = "sex"
# Define perturbations
deltas = {
    "age": 1,
    "capital.gain": 1000,
    "capital.loss": 1000,
    "hours.per.week": 1,
}


def load_and_preprocess_adult_dataset():
    """Loads and preprocesses the adult dataset."""
    df = pd.read_csv("datasets/adult.csv").dropna()
    df["income_label"] = df["income"].map({"<=50K": 0, ">50K": 1})

    categorical = [
        "workclass",
        "education",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native.country",
    ]
    for col in categorical:
        df[col] = LabelEncoder().fit_transform(df[col])

    features = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education.num",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital.gain",
        "capital.loss",
        "hours.per.week",
        "native.country",
    ]

    X = df[features]
    y = df["income_label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, features, deltas, sensitive
