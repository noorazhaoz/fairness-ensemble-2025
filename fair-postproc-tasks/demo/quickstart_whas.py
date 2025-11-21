"""
Quickstart demo for the WHAS survival dataset with a simple global
mixture optimization between a performance-oriented RSF model and a
fairness-oriented Cox model.

This demo shows:
    1. Loading and preprocessing the WHAS dataset via load_whas()
    2. Training RSF (performance model)
    3. Training CoxPH (fairness-oriented baseline using 'lenstay')
    4. Constructing survival predictions for both models
    5. Optimizing a global weight w in [0,1] to combine RSF & Cox
    6. Evaluating test C-index and a survival group fairness metric

The goal is to provide a clear, compact example of survival
post-processing with optimization. It is not intended to fully
reproduce all experiments in the paper.

Usage:
    python demo/quickstart_whas.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from lifelines import CoxPHFitter

from utils.data_loader import load_whas
from metrics.fairness import survival_group_fairness


# ---------------------------------------------------------------------
# Helper: convert to structured array if your load_whas does not
# already provide it. If your load_whas already returns structured
# arrays for scikit-survival, you can skip this and adapt accordingly.
# ---------------------------------------------------------------------
def to_structured_y(time, event):
    """
    Convert separate time and event arrays into a structured array
    required by scikit-survival.
    """
    return np.array(
        [(bool(e), float(t)) for e, t in zip(event, time)],
        dtype=[("death", "?"), ("futime", "<f8")],
    )


# ---------------------------------------------------------------------
# Survival prediction helpers
# ---------------------------------------------------------------------
def get_rsf_surv(rsf_model, Xmat, times):
    funcs = rsf_model.predict_survival_function(Xmat)
    return np.array([fn(times) for fn in funcs])  # shape [n, T]


def get_cox_surv(cox_model, Xmat, times):
    """
    Simple surrogate used in the original WHAS script:
    expectation of survival time mapped through an exponential kernel.
    """
    surv_times = cox_model.predict_expectation(Xmat[["lenstay"]]).values
    times = np.asarray(times, float)
    return np.exp(-np.outer(surv_times, 1.0 / times))


# ---------------------------------------------------------------------
# Objective for optimizing global w
# ---------------------------------------------------------------------
def objective(w_arr, rsf_model, cox_model, X_train, s_train, time_grid, lam):
    """
    Objective = L2 distance between mean RSF survival and mixed survival
                + lambda * survival_group_fairness(mix_mean, s_train)
    """
    w = float(w_arr[0])

    s_rsf = get_rsf_surv(rsf_model, X_train, time_grid)
    s_cox = get_cox_surv(cox_model, X_train, time_grid)
    s_mix = w * s_rsf + (1.0 - w) * s_cox

    rsf_mean = s_rsf.mean(axis=1)
    mix_mean = s_mix.mean(axis=1)

    l2 = np.mean((mix_mean - rsf_mean) ** 2)
    fair = survival_group_fairness(mix_mean, s_train)

    return l2 + lam * fair


# ---------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------
def main():
    # NOTE: adapt this unpacking to the actual return signature of load_whas.
    # Here we assume:
    # X_train, X_test, time_train, time_test, event_train, event_test, s_train, s_test, preproc
    (
        X_train,
        X_test,
        time_train,
        time_test,
        event_train,
        event_test,
        s_train,
        s_test,
        preproc,
    ) = load_whas()

    print("\n=== WHAS Survival Quickstart (with optimization) ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")

    # If your load_whas already returns structured y for sksurv, you can
    # directly use them instead of calling to_structured_y here.
    y_train_struct = to_structured_y(time_train, event_train)
    y_test_struct = to_structured_y(time_test, event_test)

    # Build a DataFrame version of y_train for CoxPHFitter
    y_train_df = pd.DataFrame(
        {"death": event_train.astype(int), "futime": time_train.astype(float)}
    )
    y_test_df = pd.DataFrame(
        {"death": event_test.astype(int), "futime": time_test.astype(float)}
    )

    # Evaluation time grid (following your previous code)
    t_min = np.percentile(y_test_df["futime"], 5)
    t_max = y_test_df["futime"].max()
    time_grid = np.arange(t_min, t_max, 1.5)

    # --------------------------------------------------------------
    # 1. Train RSF (performance model)
    # --------------------------------------------------------------
    rsf = RandomSurvivalForest(
        n_estimators=1000,
        min_samples_split=10,
        min_samples_leaf=15,
        n_jobs=-1,
        random_state=8,
    )
    rsf.fit(X_train, y_train_struct)
    print("RSF model trained.")

    # --------------------------------------------------------------
    # 2. Train CoxPH (fairness-oriented baseline using 'lenstay')
    # --------------------------------------------------------------
    cph = CoxPHFitter()
    cox_input = pd.concat(
        [
            pd.DataFrame(X_train, columns=preproc.feature_names_in_)[["lenstay"]].reset_index(drop=True),
            y_train_df.reset_index(drop=True),
        ],
        axis=1,
    )
    cph.fit(cox_input, duration_col="futime", event_col="death")
    print("Cox model trained.")

    # --------------------------------------------------------------
    # 3. Optimize global scalar w
    # --------------------------------------------------------------
    lambda_fair = 1.0  # demo value
    print(f"\nOptimizing mixture weight with lambda={lambda_fair} ...")

    res = minimize(
        fun=objective,
        x0=np.array([0.5]),
        args=(rsf, cph, X_train, s_train, time_grid, lambda_fair),
        bounds=[(0.0, 1.0)],
        method="L-BFGS-B",
    )
    w_opt = float(res.x[0])
    print(f"Optimal w = {w_opt:.4f}")

    # --------------------------------------------------------------
    # 4. Evaluate on test set
    # --------------------------------------------------------------
    rsf_test = get_rsf_surv(rsf, X_test, time_grid)
    cox_test = get_cox_surv(cph, pd.DataFrame(X_test, columns=preproc.feature_names_in_), time_grid)
    mix_test = w_opt * rsf_test + (1.0 - w_opt) * cox_test

    # Use negative mean survival as a simple risk score
    risk_mix = -np.log(mix_test.mean(axis=1) + 1e-10)

    cidx, _, _ = concordance_index_censored(
        y_test_struct["death"], y_test_struct["futime"], risk_mix
    )
    fair = survival_group_fairness(mix_test.mean(axis=1), s_test)

    print("\n=== Test Results ===")
    print(f"C-index:                {cidx:.4f}")
    print(f"Survival group fairness {fair:.4f}")

    print("\nWHAS quickstart demo complete.\n")


if __name__ == "__main__":
    main()
