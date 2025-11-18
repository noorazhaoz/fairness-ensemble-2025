import argparse
import numpy as np
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from utils.data_loader import load_adult
from utils.common import predict_prob
from metrics.classification import classification_metrics
from metrics.fairness import demographic_parity_diff, equalized_odds_diff

from algorithms.mixture_one_pretrained import MixtureOnePretrained
from algorithms.moe_one_pretrained import MoEOnePretrained
from algorithms.mixture_two_pretrained import MixtureTwoPretrained
from algorithms.moe_two_pretrained import MoETwoPretrained


def random_1d_projection(X, seed=0):
    rng = np.random.RandomState(seed)
    w = rng.normal(size=(X.shape[1],))
    x1d = X.dot(w) if issparse(X) else X @ w
    x1d = (x1d - x1d.mean()) / (x1d.std() + 1e-9)
    return np.asarray(x1d).ravel()


def fit_base_models(Xtr, ytr, seed=0):
    # Pretrain a performance model and a fairness-oriented baseline (LR)
    perf = RandomForestClassifier(n_estimators=1000, random_state=seed).fit(Xtr, ytr)
    fair = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    alt  = GradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
    return perf, fair, alt


def run(args):
    print("[Load] Adult dataset (binary)...")
    Xtr, Xte, ytr, yte, str_, ste_, _ = load_adult(split_seed=args.seed, test_size=0.2)

    # Hold out a validation split from training for lam/gate learning
    Xtr_sub, Xval, ytr_sub, yval, str_sub, sval = train_test_split(
        Xtr, ytr, str_, test_size=0.25, random_state=args.seed, stratify=ytr
    )

    print("[Train] Pretrained experts (for methods that need them)...")
    perf_pre, fair_pre, alt = fit_base_models(Xtr_sub, ytr_sub, seed=args.seed)

    score_fn = lambda m, X, **kw: predict_prob(m, X)  # binary default
    method = args.method.lower()
    print(f"[Method] {method}")

    if method in ["mixture1", "mixture"]:
        # One-pretrained mixture:
        # - perf model is pretrained (perf_pre)
        # - fairness model is trained inside MixtureOnePretrained
        lam_arg = args.lam
        try:
            lam_arg = float(lam_arg)
        except ValueError:
            if lam_arg != "auto":
                raise

        fair_unfitted = LogisticRegression(max_iter=1000)  # train inside
        model = MixtureOnePretrained(
            perf_model=perf_pre,
            fair_model=fair_unfitted,
            lam=lam_arg,
            mu=args.mu,
            score_fn=score_fn,
        )

        if lam_arg == "auto":
            model.fit(Xtr_sub, ytr_sub, X_val=Xval, y_val=yval, sens_val=sval)
        else:
            model.fit(Xtr_sub, ytr_sub)

        y_prob = model.predict_proba(Xte)

    elif method in ["moe1", "moe"]:
        # One-pretrained MoE:
        # - train performance model + logistic gate
        # - fairness model fixed (pretrained) = fair_pre
        # Gating features: 1D projection (can be multi-D if you want)
        Z_val = random_1d_projection(Xval, seed=args.seed).reshape(-1, 1)
        Z_te  = random_1d_projection(Xte,  seed=args.seed + 1).reshape(-1, 1)

        perf_unfitted = RandomForestClassifier(n_estimators=200, random_state=args.seed)
        moe = MoEOnePretrained(
            perf_model=perf_unfitted,    # will be fit on (Xtr_sub, ytr_sub)
            fair_model=fair_pre,         # fixed/pretrained
            score_fn=score_fn,
            use_features="Z",            # use Z for the gate
        )
        moe.fit(X_train=Xtr_sub, y_train=ytr_sub, X_val=Xval, y_val=yval, Z_val=Z_val)
        y_prob = moe.predict_proba(Xte, Z=Z_te)

    elif method == "mixture2":
        # Two-pretrained mixture: combine perf_pre and fair_pre by global λ
        model = MixtureTwoPretrained(
            perf_model=perf_pre,
            fair_model=fair_pre,
            lam=args.lam2,
            score_fn=score_fn,
        ).fit(Xtr_sub, ytr_sub)
        y_prob = model.predict_proba(Xte)

    elif method == "moe2":
        # Two-pretrained MoE with logistic gate
        Z_val = random_1d_projection(Xval, seed=args.seed).reshape(-1, 1)
        Z_te  = random_1d_projection(Xte,  seed=args.seed + 1).reshape(-1, 1)

        moe2 = MoETwoPretrained(
            expert_a=perf_pre,
            expert_b=fair_pre,
            score_fn=score_fn,
            use_features="Z",
        )
        moe2.fit(X_val=Xval, y_val=yval, Z_val=Z_val)
        y_prob = moe2.predict_proba(Xte, Z=Z_te)

    else:
        raise ValueError(f"Unknown method: {args.method}")

    perf_m = classification_metrics(yte, y_prob)
    dp = demographic_parity_diff(y_prob, ste_)
    eo = equalized_odds_diff(yte, y_prob, ste_)

    print("\n[Results]")
    for k, v in perf_m.items():
        try:
            print(f"{k:>10}: {v:.4f}")
        except Exception:
            print(f"{k:>10}: {v}")
    print(f"DP diff  : {dp:.4f}")
    print(f"EO diff  : {eo:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fair Post-Processing Runner (Task-aware)")
    parser.add_argument("--method", type=str, default="mixture1",
                        choices=["mixture1", "moe1", "mixture2", "moe2", "mixture", "moe"],
                        help="Choose a method (aliases: mixture1=mixture, moe1=moe)")
    parser.add_argument("--lam", type=str, default="0.3",
                        help="Lambda for MixtureOnePretrained (float or 'auto')")
    parser.add_argument("--mu", type=float, default=0.0,
                        help="Fairness penalty weight when lam='auto'")
    parser.add_argument("--lam2", type=float, default=0.5,
                        help="Lambda for MixtureTwoPretrained")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    run(args)
