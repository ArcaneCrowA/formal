import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import DELTAS, SENSITIVE_ATTRIBUTES


def _preprocess_adult():
    df = pd.read_csv("datasets/adult.csv").replace("?", np.nan).dropna()

    df["income_label"] = df["income"].map({"<=50K": 0, ">50K": 1})
    df = df.drop("income", axis=1)

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

    features = [col for col in df.columns if col != "income_label"]

    sensitive = SENSITIVE_ATTRIBUTES["adult"]
    deltas = DELTAS["adult"]
    target = "income_label"

    return df, features, deltas, sensitive, target


def _preprocess_german():
    df = pd.read_csv("datasets/german.csv", sep=";").dropna()

    features = [col for col in df.columns if col != "Creditability"]
    sensitive = SENSITIVE_ATTRIBUTES["german"]
    deltas = DELTAS["german"]
    target = "Creditability"

    return df, features, deltas, sensitive, target


def _preprocess_loan_data():
    df = pd.read_csv("datasets/loan_data.csv").dropna()

    df["person_gender"] = df["person_gender"].map({"female": 0, "male": 1})

    # Encode all object columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    df.dropna(inplace=True)

    features = [col for col in df.columns if col != "loan_status"]
    sensitive = "person_gender"
    deltas = {
        "person_age": 1,
        "person_income": 1000,
        "loan_amnt": 1000,
        "person_emp_exp": 1,
    }
    target = "loan_status"

    return df, features, deltas, sensitive, target


def load_and_preprocess_dataset(dataset_name: str):
    """Loads and preprocesses the specified dataset."""
    if dataset_name == "adult":
        df, features, deltas, sensitive, target = _preprocess_adult()
    elif dataset_name == "german":
        df, features, deltas, sensitive, target = _preprocess_german()
    elif dataset_name == "loan_data":
        df, features, deltas, sensitive, target = _preprocess_loan_data()
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported.")

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, features, deltas, sensitive
