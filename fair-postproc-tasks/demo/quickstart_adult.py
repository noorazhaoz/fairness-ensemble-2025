from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from utils.data_loader import load_adult
from algorithms.mixture_two_pretrained import MixtureTwoPretrained
from metrics.classification import classification_metrics
from metrics.fairness import demographic_parity_diff, equalized_odds_diff
from utils.common import predict_prob

if __name__ == "__main__":
    print("[Demo] Loading dataset...")
    Xtr, Xte, ytr, yte, str_, ste_, _ = load_adult()

    print("[Demo] Training two base experts...")
    # Performance-oriented expert (Random Forest)
    perf = RandomForestClassifier(n_estimators=100, random_state=42).fit(Xtr, ytr)
    # Fairness-oriented expert (Logistic Regression)
    fair = LogisticRegression(max_iter=500).fit(Xtr, ytr)

    print("[Demo] Combining experts using MixtureTwoPretrained (λ=0.3)...")
    model = MixtureTwoPretrained(
        perf_model=perf,
        fair_model=fair,
        lam=0.3,   # global fairness–performance weight
        score_fn=lambda m, X: predict_prob(m, X)
    ).fit(Xtr, ytr)

    print("[Demo] Evaluating mixture model...")
    y_prob = model.predict_proba(Xte)

    # Performance and fairness evaluation
    print("Performance metrics:", classification_metrics(yte, y_prob))
    print(f"DP diff: {demographic_parity_diff(y_prob, ste_):.4f}")
    print(f"EO diff: {equalized_odds_diff(yte, y_prob, ste_):.4f}")
