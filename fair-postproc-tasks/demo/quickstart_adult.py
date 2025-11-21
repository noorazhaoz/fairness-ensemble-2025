"""
Quickstart demo for running the four post-processing methods on the Adult dataset.

Supported methods:
    - mixture_one
    - mixture_two
    - moe_one
    - moe_two

Usage examples:
    python quickstart_adult.py --method mixture_two
    python quickstart_adult.py --method moe_one
"""

import argparse
import numpy as np
from sklearn.metrics import accuracy_score

# Update these imports according to your actual file structure
from utils.data_loader import load_adult
from utils.common import train_base_models, evaluate_fairness

from algorithms.mixture_one_pretrained import MixtureOnePretrained
from algorithms.mixture_two_pretrained import MixtureTwoPretrained
from algorithms.moe_one_pretrained import MoEOnePretrained
from algorithms.moe_two_pretrained import MoETwoPretrained


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Quickstart demo for Adult dataset.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["mixture_one", "mixture_two", "moe_one", "moe_two"],
        help="Choose which post-processing method to run.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Main demo logic
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    # ---------------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------------
    X_train, X_test, y_train, y_test, s_train, s_test = load_adult()

    # ---------------------------------------------------------------
    # 2. Train / load the two base models
    # ---------------------------------------------------------------
    perf_model, fair_model = train_base_models(X_train, y_train, s_train)

    y_perf_train = perf_model.predict_proba(X_train)[:, 1]
    y_perf_test = perf_model.predict_proba(X_test)[:, 1]

    y_fair_train = fair_model.predict_proba(X_train)[:, 1]
    y_fair_test = fair_model.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------------
    # 3. Select the post-processing method
    # ---------------------------------------------------------------
    if args.method == "mixture_one":
        model = MixtureOnePretrained()
    elif args.method == "mixture_two":
        model = MixtureTwoPretrained()
    elif args.method == "moe_one":
        model = MoEOnePretrained()
    elif args.method == "moe_two":
        model = MoETwoPretrained()
    else:
        raise ValueError(f"Unknown method: {args.method}")

    print(f"\nRunning method: {args.method}\n")

    # ---------------------------------------------------------------
    # 4. Fit the post-processing model
    # ---------------------------------------------------------------
    model.fit(
        y_perf_train=y_perf_train,
        y_fair_train=y_fair_train,
        y_true_train=y_train,
        sensitive_train=s_train,
    )

    # ---------------------------------------------------------------
    # 5. Test-time predictions
    # ---------------------------------------------------------------
    y_pred_prob = model.predict_proba(
        y_perf_test=y_perf_test,
        y_fair_test=y_fair_test,
        sensitive_test=s_test,
    )

    y_pred_label = (y_pred_prob >= 0.5).astype(int)

    # ---------------------------------------------------------------
    # 6. Evaluate accuracy and fairness
    # ---------------------------------------------------------------
    acc = accuracy_score(y_test, y_pred_label)
    dp_diff, eo_diff = evaluate_fairness(
        y_true=y_test,
        y_pred_prob=y_pred_prob,
        sensitive=s_test,
    )

    # ---------------------------------------------------------------
    # 7. Print results
    # ---------------------------------------------------------------
    print("=== Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Demographic Parity Difference: {dp_diff:.4f}")
    print(f"Equalized Odds Difference:     {eo_diff:.4f}\n")


if __name__ == "__main__":
    main()
