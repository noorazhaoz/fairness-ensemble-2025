"""
Quickstart demo for the Adult dataset (one-pretrained Mixture model).

This script:
    - Loads a preprocessed Adult dataset from ../data/adult.csv
    - Trains a performance model (Random Forest) on all features
    - Uses a simple one-feature post-processing head (Logistic Regression
      on "education-num") as the fairness-oriented component
    - Optimizes a global scalar weight w in [0, 1] together with the
      post-processing head parameters to minimize

          L = CE(y_perf, y_mix)/n + lamda * DP(y_mix >= 0.5),

      where y_mix = w * y_perf + (1 - w) * y_pp.

    - Evaluates accuracy, demographic parity difference, and equalized
      odds difference on the test set.

This demo is a simplified “one-pretrained Mixture” example and is not
intended to fully reproduce the full experimental pipeline.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy.optimize as opt

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)


# ----------------------------------------------------------------------
# Cross-entropy loss between two probability vectors
# Treat y_perf as the "soft target" and y_mix as the prediction
# ----------------------------------------------------------------------
def cross_entropy_loss(perf_probs, mixed_probs):
    epsilon = 1e-16
    perf_probs = np.clip(perf_probs, epsilon, 1 - epsilon)
    mixed_probs = np.clip(mixed_probs, epsilon, 1 - epsilon)
    return -np.sum(
        perf_probs * np.log(mixed_probs)
        + (1 - perf_probs) * np.log(1 - mixed_probs)
    )


# ----------------------------------------------------------------------
# Post-processing head: logistic function on "education-num"
# ----------------------------------------------------------------------
def pp_model_output(X, weights, bias):
    """
    Compute the output of a simple 1-feature logistic head using
    "education-num" as input feature.
    """
    X_edu = X[["education-num"]]
    logits = np.dot(X_edu, weights) + bias
    return 1.0 / (1.0 + np.exp(-logits))


# ----------------------------------------------------------------------
# Optimize the post-processing head parameters (weights, bias)
# for a fixed mixture weight w
# ----------------------------------------------------------------------
def optimize_logreg_params(lamda, X_train, y_perf_train, w, s_train):
    """
    Optimize the parameters (weights, bias) of the post-processing head
    for a fixed global mixture weight w.

    Objective:
        CE(y_perf, y_mix)/n + lamda * DP(y_mix >= 0.5)

    where:
        y_mix = w * y_perf + (1 - w) * y_pp
    """
    n_features = X_train[["education-num"]].shape[1]
    init_params = np.zeros(n_features + 1)  # weights + bias

    def objective(params):
        weights = params[:-1]
        bias = params[-1]

        pp_probs = pp_model_output(X_train, weights, bias)
        rf_probs = y_perf_train  # shape [n]

        y_mix = w * rf_probs + (1.0 - w) * pp_probs
        y_pred = (y_mix >= 0.5).astype(int)

        ce = cross_entropy_loss(rf_probs, y_mix)
        dp = demographic_parity_difference(
            y_true=None,  # will be ignored if using predictions only
            y_pred=y_pred,
            sensitive_features=s_train,
        )

        # Fairlearn's DP usually takes y_true, but for a simple demo we
        # follow a prediction-only style, so you may also replace this
        # with a custom DP implementation if preferred.
        return ce / len(X_train) + lamda * dp

    result = opt.minimize(objective, init_params, method="L-BFGS-B")
    return result.x[:-1], result.x[-1]


# ----------------------------------------------------------------------
# Optimize the global mixture weight w for fixed head parameters
# ----------------------------------------------------------------------
def optimize_weight(lamda, X_train, y_perf_train, weights, bias, s_train):
    """
    Optimize the global mixture weight w in [0, 1] for fixed
    post-processing head parameters (weights, bias).
    """

    pp_probs = pp_model_output(X_train, weights, bias)
    rf_probs = y_perf_train

    def objective(w):
        w = float(w)
        y_mix = w * rf_probs + (1.0 - w) * pp_probs
        y_pred = (y_mix >= 0.5).astype(int)

        ce = cross_entropy_loss(rf_probs, y_mix)
        dp = demographic_parity_difference(
            y_true=None,  # see note above
            y_pred=y_pred,
            sensitive_features=s_train,
        )
        return ce / len(X_train) + lamda * dp

    result = opt.minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    return float(result.x)


# ----------------------------------------------------------------------
# Main quickstart demo
# ----------------------------------------------------------------------
def main():
    # ==============================================================
    # 1. Load and preprocess Adult dataset
    # ==============================================================
    df = pd.read_csv("../data/adult.csv")

    X = df.drop(columns="income")
    y = df["income"].values
    s = df["gender"].values

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X,
        y,
        s,
        test_size=0.2,
        random_state=42,
        stratify=s,
    )

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns,
        index=X_test.index,
    )

    # ==============================================================
    # 2. Performance model: Random Forest on all features
    # ==============================================================
    rf_perf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_perf.fit(X_train_scaled, y_train)

    y_perf_train = rf_perf.predict_proba(X_train_scaled)[:, 1]
    y_perf_test = rf_perf.predict_proba(X_test_scaled)[:, 1]

    # Baseline for reference
    baseline_acc = accuracy_score(y_test, rf_perf.predict(X_test_scaled))
    print(f"\n📊 Baseline RF accuracy (test): {baseline_acc:.4f}\n")

    # ==============================================================
    # 3. One-pretrained Mixture optimization over lamda
    # ==============================================================
    lambda_values = [0.01, 0.05, 0.1, 1, 5, 10]
    all_results = []

    for lamda in lambda_values:
        print(f"=== lamda = {lamda} ===")

        # Initialize w and head parameters
        w = 0.5
        prev_obj = float("inf")

        # Simple alternating optimization: head params ↔ global weight
        for _ in range(5):
            # Step 1: optimize head for fixed w
            opt_w_vec, opt_b = optimize_logreg_params(
                lamda, X_train_scaled, y_perf_train, w, s_train
            )

            # Step 2: optimize w for fixed head
            w_new = optimize_weight(
                lamda, X_train_scaled, y_perf_train, opt_w_vec, opt_b, s_train
            )

            # Compute objective after update
            pp_probs = pp_model_output(X_train_scaled, opt_w_vec, opt_b)
            y_mix_train = w_new * y_perf_train + (1.0 - w_new) * pp_probs
            y_pred_train = (y_mix_train >= 0.5).astype(int)

            ce = cross_entropy_loss(y_perf_train, y_mix_train)
            dp = demographic_parity_difference(
                y_true=None,
                y_pred=y_pred_train,
                sensitive_features=s_train,
            )
            obj = ce / len(X_train_scaled) + lamda * dp

            # Check convergence
            if abs(obj - prev_obj) < 1e-4:
                w = w_new
                prev_obj = obj
                break

            w = w_new
            prev_obj = obj

        # ==========================================================
        # Evaluate on test set with (w, opt_w_vec, opt_b)
        # ==========================================================
        y_pp_test = pp_model_output(X_test_scaled, opt_w_vec, opt_b)
        y_mix_test = w * y_perf_test + (1.0 - w) * y_pp_test
        y_pred_test = (y_mix_test >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred_test)
        dp_test = demographic_parity_difference(
            y_true=y_test,
            y_pred=y_pred_test,
            sensitive_features=s_test,
        )
        eo_test = equalized_odds_difference(
            y_true=y_test,
            y_pred=y_pred_test,
            sensitive_features=s_test,
        )

        print(
            f"lamda={lamda:<6} | "
            f"w*={w:.4f} | "
            f"Acc={acc:.4f} | DP={dp_test:.4f} | EO={eo_test:.4f} | Obj={prev_obj:.4f}"
        )

        all_results.append(
            {
                "lamda": lamda,
                "w_opt": w,
                "Accuracy": acc,
                "Fairness_DP": dp_test,
                "Fairness_EO": eo_test,
                "Objective": prev_obj,
            }
        )

    # Save results (optional)
    results_df = pd.DataFrame(all_results)
    results_df.to_excel("../output/adult_one_pretrained_mixture_demo.xlsx", index=False)
    print("\nResults saved to ../output/adult_one_pretrained_mixture_demo.xlsx")
    print("Adult one-pretrained Mixture quickstart demo complete.\n")


if __name__ == "__main__":
    main()
