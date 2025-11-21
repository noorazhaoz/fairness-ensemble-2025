"""
Quickstart demo for the Adult dataset (two-pretrained MoE-style model).

This script:
    - Loads a preprocessed Adult dataset from ../data/adult.csv
    - Trains a performance model (Random Forest) on all features
    - Trains a fairness-oriented model (Logistic Regression on a single feature)
    - Trains an instance-wise weight regressor (Logistic Regression)
    - Optimizes the weight regressor parameters via:
          cross-entropy(performance_probs, weighted_probs)
          + lamda * demographic_parity_difference
    - Evaluates accuracy, demographic parity, and equalized odds on the test set

The goal is to illustrate a dynamic (instance-wise) mixture (MoE-style)
between a strong model and a simpler, fairness-oriented model.
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
# (perf_probs are treated as "soft targets" because this is post-processing)
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
# Post-processing fairness model outputs with temperature scaling
# ----------------------------------------------------------------------
def pp_model_output(X, weights, bias, temperature=2.0):
    logits = np.dot(X, weights) + bias
    logits /= temperature
    return 1.0 / (1.0 + np.exp(-logits))


# ----------------------------------------------------------------------
# Weight regressor (Logistic Regression in weight space)
# ----------------------------------------------------------------------
class WeightRegressor:
    def __init__(self):
        # We will set coef_ and intercept_ manually during optimization
        self.model = LogisticRegression(
            penalty="l2", C=10.0, random_state=42
        )

    def predict_weights(self, X):
        raw = np.dot(X, self.model.coef_.T) + self.model.intercept_
        return 1.0 / (1.0 + np.exp(-raw))


# ----------------------------------------------------------------------
# Main demo
# ----------------------------------------------------------------------
def main():
    # ==============================================================
    # 1. Load dataset
    #    We assume ../data/adult.csv exists with columns:
    #      - 'income' (target, 0/1)
    #      - 'gender' (sensitive attribute, 0/1)
    #      - other feature columns
    # ==============================================================
    df = pd.read_csv("../data/adult.csv")

    X = df.drop(columns="income")
    y = df["income"].values
    sensitive_attr = df["gender"].values

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X,
        y,
        sensitive_attr,
        test_size=0.2,
        random_state=42,
        stratify=sensitive_attr,
    )

    # Scaling
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
    # 3. Fairness-oriented model: Logistic Regression on a single feature
    #    (here we use 'education-num' as in your original code)
    # ==============================================================
    fair_features = ["education-num"]
    lr_fair = LogisticRegression()
    lr_fair.fit(X_train_scaled[fair_features], y_train)

    # We use a temperature-scaled version of the fairness model's output
    y_fair_train = pp_model_output(
        X_train_scaled[fair_features],
        lr_fair.coef_.ravel(),
        lr_fair.intercept_,
        temperature=2.0,
    )
    y_fair_test = pp_model_output(
        X_test_scaled[fair_features],
        lr_fair.coef_.ravel(),
        lr_fair.intercept_,
        temperature=2.0,
    )

    # ==============================================================
    # 4. Weight regressor (instance-wise MoE weights)
    # ==============================================================
    weight_regressor = WeightRegressor()

    # Objective with dynamic (instance-wise) weights
    def objective_with_dynamic_weights(params, lamda, X_input, perf_probs, fair_probs, y_true, s_sensitive):
        # Set the logistic regression parameters manually
        weight_regressor.model.coef_ = np.array([params[:-1]])
        weight_regressor.model.intercept_ = np.array([params[-1]])

        # Instance-wise weights in [0,1]
        weights = weight_regressor.predict_weights(X_input).ravel()

        # Mixture of performance + fairness model
        weighted_probs = weights * perf_probs + (1.0 - weights) * fair_probs
        y_pred_binary = (weighted_probs >= 0.5).astype(int)

        # Cross-entropy (using perf_probs as soft targets) + fairness penalty
        ce = cross_entropy_loss(perf_probs, weighted_probs)

        # Group fairness: demographic parity difference
        dp = demographic_parity_difference(
            y_true,
            y_pred_binary,
            sensitive_features=s_sensitive,
        )

        # Average objective per instance
        return ce / len(X_input) + lamda * dp

    # ==============================================================
    # 5. Experiment loop over lamda
    # ==============================================================
    lambda_values = [0.01, 0.05, 0.1, 1, 5, 10, 50, 100]
    all_results = []

    # Optional: baseline sanity check
    baseline_perf_acc = accuracy_score(y_test, rf_perf.predict(X_test_scaled))
    baseline_fair_acc = accuracy_score(
        y_test, lr_fair.predict(X_test_scaled[fair_features])
    )
    print(f"\n📊 Baseline RF (performance) accuracy: {baseline_perf_acc:.4f}")
    print(f"📊 Baseline LR (fairness) accuracy:   {baseline_fair_acc:.4f}\n")

    for lamda in lambda_values:
        best_obj = float("inf")
        best_params = None

        # You can run multiple random initializations if needed; here we keep 1 for demo
        for _ in range(1):
            init_params = np.random.randn(X_train_scaled.shape[1] + 1)
            result = opt.minimize(
                fun=objective_with_dynamic_weights,
                x0=init_params,
                args=(lamda, X_train_scaled, y_perf_train, y_fair_train, y_train, s_train),
                method="L-BFGS-B",
                options={"maxiter": 500},
            )

            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x

        # Set best parameters
        weight_regressor.model.coef_ = np.array([best_params[:-1]])
        weight_regressor.model.intercept_ = np.array([best_params[-1]])

        # Instance-wise weights on test
        weights_test = weight_regressor.predict_weights(X_test_scaled).ravel()

        # Final mixture on test
        y_weighted_test = weights_test * y_perf_test + (1.0 - weights_test) * y_fair_test
        y_pred_final = (y_weighted_test >= 0.5).astype(int)

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
            f"✅ lamda={lamda:<6} | "
            f"Acc={acc:.4f} | DP={dp:.4f} | EO={eo:.4f} | Mean(w)={np.mean(weights_test):.3f}"
        )

        all_results.append(
            {
                "lamda": lamda,
                "Accuracy": acc,
                "Fairness_DP": dp,
                "Fairness_EO": eo,
                "Mean Weight": np.mean(weights_test),
                "Weight Variance": np.var(weights_test),
            }
        )

    # ==============================================================
    # 6. Save results (optional)
    # ==============================================================
    df_results = pd.DataFrame(all_results)
    df_results.to_excel("../output/adult_two_pretrained_moe_demo.xlsx", index=False)
    print("\nResults saved to ../output/adult_two_pretrained_moe_demo.xlsx")
    print("Adult MoE quickstart demo complete.\n")


if __name__ == "__main__":
    main()
