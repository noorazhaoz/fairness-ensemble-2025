"""
Quickstart demo for the Adult dataset (two-pretrained Mixture model).

This script:
    - Loads a preprocessed Adult dataset from ../data/adult.csv
    - Trains a performance model (Random Forest) on all features
    - Trains a fairness-oriented model (Logistic Regression on 'education-num')
    - Optimizes a single global weight w in [0, 1] to combine the two models:
          y_mix = w * y_perf + (1 - w) * y_fair
    - Uses the objective:
          cross_entropy(y_perf, y_mix) / n + lamda * demographic_parity_difference
    - Evaluates accuracy, demographic parity, and equalized odds on the test set

This provides a simple “two-pretrained Mixture” counterpart to the
instance-wise MoE demo.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import scipy.optimize as opt


# ----------------------------------------------------------------------
# Cross-entropy loss between two probability vectors
# ----------------------------------------------------------------------
def cross_entropy_loss(perf_probs, weighted_probs):
    epsilon = 1e-16
    perf_probs = np.clip(perf_probs, epsilon, 1 - epsilon)
    weighted_probs = np.clip(weighted_probs, epsilon, 1 - epsilon)
    return -np.sum(
        perf_probs * np.log(weighted_probs)
        + (1 - perf_probs) * np.log(1 - weighted_probs)
    )


# ----------------------------------------------------------------------
# Main demo
# ----------------------------------------------------------------------
def main():
    # ==============================================================
    # 1. Load dataset
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

    # ==============================================================
    # 3. Fairness-oriented model: Logistic Regression on 'education-num'
    # ==============================================================
    fair_features = ["education-num"]

    lr_fair = LogisticRegression()
    lr_fair.fit(X_train_scaled[fair_features], y_train)

    y_fair_train = lr_fair.predict_proba(X_train_scaled[fair_features])[:, 1]
    y_fair_test = lr_fair.predict_proba(X_test_scaled[fair_features])[:, 1]

    # ==============================================================
    # 4. Global mixture weight optimization
    # ==============================================================

    def objective_global(w_arr, lamda, perf_probs, fair_probs, y_true, s_sensitive):
        """
        Objective for a single global weight w:
            y_mix = w * perf_probs + (1 - w) * fair_probs

            L(w) = CE(perf_probs, y_mix)/n + lamda * DP(y_true, y_mix >= 0.5)
        """
        w = float(w_arr[0])

        # Mixture probabilities
        y_mix = w * perf_probs + (1.0 - w) * fair_probs
        y_pred = (y_mix >= 0.5).astype(int)

        # Cross-entropy with perf_probs as soft targets
        ce = cross_entropy_loss(perf_probs, y_mix)

        # Demographic parity difference as fairness term
        dp = demographic_parity_difference(
            y_true,
            y_pred,
            sensitive_features=s_sensitive,
        )

        return ce / len(y_true) + lamda * dp

    lambda_values = [0.01, 0.05, 0.1, 1, 5, 10, 50, 100]
    all_results = []

    baseline_perf_acc = accuracy_score(y_test, rf_perf.predict(X_test_scaled))
    baseline_fair_acc = accuracy_score(
        y_test, lr_fair.predict(X_test_scaled[fair_features])
    )

    print(f"\n📊 Baseline RF (performance) accuracy: {baseline_perf_acc:.4f}")
    print(f"📊 Baseline LR (fairness) accuracy:   {baseline_fair_acc:.4f}\n")

    for lamda in lambda_values:
        init_w = np.array([0.5])

        res = opt.minimize(
            fun=objective_global,
            x0=init_w,
            args=(lamda, y_perf_train, y_fair_train, y_train, s_train),
            bounds=[(0.0, 1.0)],
            method="L-BFGS-B",
            options={"maxiter": 500},
        )

        best_w = float(res.x[0])
        best_obj = res.fun

        # Evaluate on test set with the optimized w
        y_mix_test = best_w * y_perf_test + (1.0 - best_w) * y_fair_test
        y_pred_final = (y_mix_test >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred_final)
        dp = demographic_parity_difference(
            y_test,
            y_pred_final,
            sensitive_features=s_test,
        )
        eo = equalized_odds_difference(
            y_test,
            y_pred_final,
            sensitive_features=s_test,
        )

        print(
            f"✅ lamda={lamda:<6} | w*={best_w:.4f} | "
            f"Acc={acc:.4f} | DP={dp:.4f} | EO={eo:.4f} | Obj={best_obj:.4f}"
        )

        all_results.append(
            {
                "lamda": lamda,
                "w_opt": best_w,
                "Accuracy": acc,
                "Fairness_DP": dp,
                "Fairness_EO": eo,
                "Objective": best_obj,
            }
        )

    df_results = pd.DataFrame(all_results)
    df_results.to_excel("../output/adult_two_pretrained_mixture_demo.xlsx", index=False)
    print("\nResults saved to ../output/adult_two_pretrained_mixture_demo.xlsx")
    print("Adult two-pretrained Mixture quickstart demo complete.\n")


if __name__ == "__main__":
    main()
