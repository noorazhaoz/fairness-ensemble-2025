"""
Adult dataset — one-pretrained MoE (RF + simple post-processing head).

This script implements:
    - Performance model: Random Forest trained on all features
    - Fairness head: a simple one-feature logistic head on "education-num"
    - Instance-wise weights: a logistic weight regressor over all features
    - Alternating optimization between:
        (1) the fairness head parameters (f_params, f_bias)
        (2) the weight regressor parameters (w_params, w_bias)
    - Objective:
        L = CE(weighted_probs, perf_probs) / n + lamda * DP_custom(preds, sensitive)

      where
        weighted_probs = w(x) * y_perf + (1 - w(x)) * y_fair.

The script:
    - Uses bootstrap on the training set to learn parameters
    - Uses 30 bootstrap samples of the test set to estimate performance
    - Computes accuracy, Demographic Parity and Equalized Odds using
      fairlearn.metrics on the test set
    - Computes 95% confidence intervals for each lambda and metric
    - Saves full results and CI to Excel files in ../output/

This is the “one-pretrained MoE” counterpart of the other Adult demos.
"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Cross-entropy loss between two probability vectors
# Here we treat weighted_probs as "prediction" and perf_probs as target.
# ----------------------------------------------------------------------
def cross_entropy_loss(weighted_probs, perf_probs):
    epsilon = 1e-16
    weighted_probs = np.clip(weighted_probs, epsilon, 1 - epsilon)
    perf_probs = np.clip(perf_probs, epsilon, 1 - epsilon)
    return -np.sum(
        weighted_probs * np.log(perf_probs)
        + (1 - weighted_probs) * np.log(1 - perf_probs)
    )


# ----------------------------------------------------------------------
# Custom demographic parity difference used ONLY inside the objective
# (prediction-only version, no y_true needed)
# ----------------------------------------------------------------------
def demographic_parity_difference_custom(predictions, sensitive_features):
    predictions = np.asarray(predictions)
    sensitive_features = np.asarray(sensitive_features)
    return float(
        np.abs(
            np.mean(predictions[sensitive_features == 1])
            - np.mean(predictions[sensitive_features == 0])
        )
    )


# ----------------------------------------------------------------------
# Confidence interval helper
# ----------------------------------------------------------------------
def compute_confidence_interval(data):
    data = np.asarray(data)
    mean_val = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(len(data))
    ci_lower, ci_upper = norm.interval(0.95, loc=mean_val, scale=std_err)
    return mean_val, ci_lower, ci_upper


# ----------------------------------------------------------------------
# Weight regressor (instance-wise gating)
# ----------------------------------------------------------------------
class WeightRegressor:
    def __init__(self):
        self.model = LogisticRegression(
            penalty="l2", C=10.0, random_state=42
        )

    def predict_weights(self, X):
        # If uninitialized, default to 0.5 for all points
        if not hasattr(self.model, "coef_"):
            return np.ones(len(X)) * 0.5

        raw = np.dot(X, self.model.coef_.T) + self.model.intercept_
        weights = 1.0 / (1.0 + np.exp(-raw))
        return np.clip(weights, 0.0, 1.0).ravel()

    def update_weights(self, new_params):
        self.model.coef_ = np.array([new_params[:-1]])
        self.model.intercept_ = np.array([new_params[-1]])


# ----------------------------------------------------------------------
# Fairness model output: simple logistic head on "education-num"
# ----------------------------------------------------------------------
def pp_model_output(X, fairness_params, fairness_bias):
    """
    One-feature fairness head based on 'education-num'.
    """
    X_edu = X[["education-num"]]
    logits = np.dot(X_edu, fairness_params) + fairness_bias
    return 1.0 / (1.0 + np.exp(-logits))


# ----------------------------------------------------------------------
# Fairness head objective (for fixed weights)
# ----------------------------------------------------------------------
def fairness_objective(params, lamda, Xb, perf_probs, weights, sensitive_feature_name):
    """
    Optimize fairness head parameters (fairness_params, fairness_bias)
    for fixed instance-wise weights.

    Objective:
        L = CE(weighted_probs, perf_probs) / n
            + lamda * DP_custom(preds, sensitive)
    """
    fairness_params = params[:-1]
    fairness_bias = params[-1]

    fairness_probs = pp_model_output(Xb, fairness_params, fairness_bias)
    weighted_probs = weights * perf_probs + (1.0 - weights) * fairness_probs

    loss = cross_entropy_loss(weighted_probs, perf_probs) / len(Xb)
    preds = (weighted_probs >= 0.5).astype(int)

    fairness = demographic_parity_difference_custom(
        preds,
        Xb[sensitive_feature_name].values,
    )
    return loss + lamda * fairness


# ----------------------------------------------------------------------
# Weight regressor objective (for fixed fairness head)
# ----------------------------------------------------------------------
def weight_objective(
    params,
    lamda,
    Xb,
    perf_probs,
    fairness_params,
    fairness_bias,
    regressor,
    sensitive_feature_name,
):
    """
    Optimize weight regressor parameters (w_params, w_bias) for fixed
    fairness head parameters.

    Objective:
        L = CE(weighted_probs, perf_probs) / n
            + lamda * DP_custom(preds, sensitive)
    """
    regressor.update_weights(params)
    weights = regressor.predict_weights(Xb)

    fairness_probs = pp_model_output(Xb, fairness_params, fairness_bias)
    weighted_probs = weights * perf_probs + (1.0 - weights) * fairness_probs

    loss = cross_entropy_loss(weighted_probs, perf_probs) / len(Xb)
    preds = (weighted_probs >= 0.5).astype(int)

    fairness = demographic_parity_difference_custom(
        preds,
        Xb[sensitive_feature_name].values,
    )
    return loss + lamda * fairness


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------
def main():
    sensitive_feature = "gender"

    # === Load and preprocess Adult dataset ===
    df = pd.read_csv("../data/adult.csv")
    X = df.drop(columns="income")
    y = df["income"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=df[sensitive_feature],
    )

    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns
    )

    # === Train performance model ===
    rf_perf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_perf.fit(X_train_scaled, y_train)

    print("Performance model (Random Forest) trained.")

    # === Hyperparameters ===
    lambda_values = [0.01, 0.05, 1, 5, 10, 100, 500]
    num_train_bootstrap = 1
    num_test_bootstrap = 30

    results_all = []

    # === Train-time bootstrap ===
    for i in range(num_train_bootstrap):
        print(f"\n=== Train bootstrap {i} ===")
        Xb_train, yb_train = resample(
            X_train_scaled, y_train, random_state=i
        )
        perf_probs = rf_perf.predict_proba(Xb_train)[:, 1]

        for lamda in lambda_values:
            print(f"--- lamda = {lamda} ---")
            weight_regressor = WeightRegressor()
            best_objective = float("inf")
            best_w = None
            best_f_params = None
            best_f_bias = None

            # One random initialization (can increase if needed)
            w_params = np.random.randn(X_train.shape[1])
            w_bias = np.random.randn()
            f_params = np.random.randn(1)
            f_bias = np.random.randn()

            # Alternating optimization: fairness head ↔ weight regressor
            for _ in range(3):
                # 1) Optimize fairness head for current weights
                weights_current = weight_regressor.predict_weights(Xb_train)
                fairness_opt = minimize(
                    fairness_objective,
                    np.append(f_params, f_bias),
                    args=(
                        lamda,
                        Xb_train,
                        perf_probs,
                        weights_current,
                        sensitive_feature,
                    ),
                    method="L-BFGS-B",
                )
                f_params = fairness_opt.x[:-1]
                f_bias = fairness_opt.x[-1]

                # 2) Optimize weight regressor for fixed fairness head
                weight_opt = minimize(
                    weight_objective,
                    np.append(w_params, w_bias),
                    args=(
                        lamda,
                        Xb_train,
                        perf_probs,
                        f_params,
                        f_bias,
                        weight_regressor,
                        sensitive_feature,
                    ),
                    method="L-BFGS-B",
                )

                if weight_opt.fun < best_objective:
                    best_objective = weight_opt.fun
                    best_w = weight_opt.x
                    best_f_params = f_params.copy()
                    best_f_bias = float(f_bias)

                # Update params for next alternating step
                w_params = weight_opt.x[:-1]
                w_bias = weight_opt.x[-1]

            # Fix best parameters
            weight_regressor.update_weights(best_w)

            # === Test-time bootstrap ===
            for j in range(num_test_bootstrap):
                Xt, yt = resample(X_test_scaled, y_test, random_state=j)
                sens_t = Xt[sensitive_feature].values

                weights_t = weight_regressor.predict_weights(Xt)
                fairness_probs = pp_model_output(Xt, best_f_params, best_f_bias)
                perf_probs_test = rf_perf.predict_proba(Xt)[:, 1]
                weighted_probs = (
                    weights_t * perf_probs_test
                    + (1.0 - weights_t) * fairness_probs
                )
                preds = (weighted_probs >= 0.5).astype(int)

                # t-SNE visualization only for the first test bootstrap
                if j == 0:
                    tsne = TSNE(
                        n_components=2,
                        perplexity=30,
                        n_iter=1000,
                        random_state=42,
                    )
                    Xt_2d = tsne.fit_transform(Xt)

                    plt.figure(figsize=(10, 8))
                    scatter = plt.scatter(
                        Xt_2d[:, 0],
                        Xt_2d[:, 1],
                        c=weights_t,
                        cmap="viridis",
                        s=50,
                        alpha=0.8,
                    )
                    plt.colorbar(scatter, label="Instance-Level Weights")
                    plt.title(
                        "t-SNE Visualization of Test Set with Weights",
                        fontsize=16,
                    )
                    plt.xlabel("t-SNE Dimension 1")
                    plt.ylabel("t-SNE Dimension 2")
                    plt.show()

                    # Group-wise visualization by sensitive attribute
                    Xt_0 = Xt[sens_t == 0]
                    Xt_1 = Xt[sens_t == 1]
                    weights_0 = weights_t[sens_t == 0]
                    weights_1 = weights_t[sens_t == 1]

                    if len(Xt_0) > 1 and len(Xt_1) > 1:
                        tsne_0 = TSNE(
                            n_components=2,
                            perplexity=30,
                            n_iter=1000,
                            random_state=42,
                        )
                        tsne_1 = TSNE(
                            n_components=2,
                            perplexity=30,
                            n_iter=1000,
                            random_state=42,
                        )
                        Xt_0_2d = tsne_0.fit_transform(Xt_0)
                        Xt_1_2d = tsne_1.fit_transform(Xt_1)

                        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

                        scatter_0 = axes[0].scatter(
                            Xt_0_2d[:, 0],
                            Xt_0_2d[:, 1],
                            c=weights_0,
                            cmap="viridis",
                            s=50,
                            alpha=0.8,
                        )
                        fig.colorbar(
                            scatter_0, ax=axes[0], label="Weights"
                        )
                        axes[0].set_title("Sensitive Attribute = 0")

                        scatter_1 = axes[1].scatter(
                            Xt_1_2d[:, 0],
                            Xt_1_2d[:, 1],
                            c=weights_1,
                            cmap="viridis",
                            s=50,
                            alpha=0.8,
                        )
                        fig.colorbar(
                            scatter_1, ax=axes[1], label="Weights"
                        )
                        axes[1].set_title("Sensitive Attribute = 1")
                        plt.tight_layout()
                        plt.show()

                # === Evaluation metrics on this bootstrap ===
                dp_fairlearn = demographic_parity_difference(
                    y_true=yt,
                    y_pred=preds,
                    sensitive_features=sens_t,
                )
                eo_fairlearn = equalized_odds_difference(
                    y_true=yt,
                    y_pred=preds,
                    sensitive_features=sens_t,
                )

                results_all.append(
                    {
                        "TrainBootstrap": i,
                        "TestBootstrap": j,
                        "Lambda": lamda,
                        "Accuracy": accuracy_score(yt, preds),
                        "Fairness_DP": dp_fairlearn,
                        "Fairness_EO": eo_fairlearn,
                    }
                )

    # === Aggregate results & confidence intervals ===
    results_df = pd.DataFrame(results_all)
    ci_rows = []

    for lam in lambda_values:
        subset = results_df[results_df["Lambda"] == lam]
        for metric in ["Accuracy", "Fairness_DP", "Fairness_EO"]:
            mean, lower, upper = compute_confidence_interval(subset[metric])
            ci_rows.append(
                {
                    "Lambda": lam,
                    "Metric": metric,
                    "Mean": mean,
                    "95% CI Lower": lower,
                    "95% CI Upper": upper,
                }
            )

    ci_df = pd.DataFrame(ci_rows)

    # === Save results ===
    results_df.to_excel("../output/adult_pp_moe_full.xlsx", index=False)
    ci_df.to_excel("../output/adult_pp_moe_CI.xlsx", index=False)
    print("✅ MLP + simp_pp MoE with EO & CI computation complete.")


if __name__ == "__main__":
    main()
