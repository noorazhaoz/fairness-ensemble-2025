import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml


def load_adult(split_seed=42, test_size=0.2):
    """Load and preprocess the Adult dataset with MinMax scaling for numeric columns."""
    data = fetch_openml(name="adult", version=2, as_frame=True)
    X, y = data.data.copy(), (data.target == ">50K").astype(int).values
    sens = (X["sex"] == "Male").astype(int).values

    # Drop fnlwgt (not meaningful for prediction)
    if "fnlwgt" in X.columns:
        X = X.drop(columns=["fnlwgt"])

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    preproc = ColumnTransformer([
        ("num", MinMaxScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    Xtr, Xte, ytr, yte, str_, ste_ = train_test_split(
        X, y, sens, test_size=test_size, random_state=split_seed, stratify=y
    )

    pipe = Pipeline([("pre", preproc)])
    Xtr_t = pipe.fit_transform(Xtr)
    Xte_t = pipe.transform(Xte)

    return Xtr_t, Xte_t, ytr, yte, str_, ste_, pipe


def load_whas(csv_path="data/whas.csv", split_seed=10, test_size=0.2):
    """
    Load and preprocess the WHAS survival dataset.

    This follows the same preprocessing logic as the existing WHAS code:
    - outcome: status (event), time (follow-up time)
    - features: ['age', 'chf', 'cpk', 'lenstay', 'miord', 'mitype', 'sexF', 'sho']
    - categorical features: ['chf', 'miord', 'mitype', 'sho'] (one-hot encoded)
    - sensitive attribute: derived from 'sexF' (renamed to 'sex')

    Returns
    -------
    X_train_t : array-like, transformed training features
    X_test_t  : array-like, transformed test features
    time_train : 1D array of survival times for train set
    time_test  : 1D array of survival times for test set
    event_train: 1D array of event indicators for train set
    event_test : 1D array of event indicators for test set
    s_train    : 1D array of sensitive attribute for train set
    s_test     : 1D array of sensitive attribute for test set
    preproc    : fitted preprocessing pipeline (MinMax scaling)
    """
    df = pd.read_csv(csv_path)

    # Build (death, futime) as in your original code
    y = df[["status", "time"]].copy()
    y = y.rename(columns={"status": "death", "time": "futime"})

    # Features (following your original 'num_column' list)
    num_columns = ["age", "chf", "cpk", "lenstay", "miord",
                   "mitype", "sexF", "sho"]
    X = df[num_columns].copy().rename(columns={"sexF": "sex"})

    # One-hot encode selected categorical features
    categorical_features = ["chf", "miord", "mitype", "sho"]
    for feature in categorical_features:
        onehot = pd.get_dummies(X[feature], prefix=feature)
        X = X.drop(feature, axis=1)
        X = X.join(onehot)

    # Sensitive attribute (based on 'sex' after renaming from 'sexF')
    # Adapt the type if needed (0/1, etc.)
    s = X["sex"].astype(int).values

    # Train-test split
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, s, test_size=test_size, random_state=split_seed
    )

    # Split outcome into time and event arrays
    time_train = y_train["futime"].values.astype(float)
    time_test = y_test["futime"].values.astype(float)
    event_train = y_train["death"].values.astype(int)
    event_test = y_test["death"].values.astype(int)

    # Optional: MinMax scaling over all numeric columns (X is already numeric)
    preproc = Pipeline([("scale", MinMaxScaler())])
    X_train_t = preproc.fit_transform(X_train)
    X_test_t = preproc.transform(X_test)

    return (
        X_train_t,
        X_test_t,
        time_train,
        time_test,
        event_train,
        event_test,
        s_train,
        s_test,
        preproc,
    )

