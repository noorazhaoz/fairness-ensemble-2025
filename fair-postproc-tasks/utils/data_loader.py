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
