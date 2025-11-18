from typing import Optional, Callable, Any, Dict
import numpy as np
from sklearn.linear_model import LogisticRegression

from utils.gating import LinearGating  # kept for compatibility if you still want it
from utils.common import predict_prob


def _default_binary_loss(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """
    Per-sample log loss for binary classification.
    y_true in {0,1}, y_score in [0,1].
    Returns an array of shape (n_samples,).
    """
    eps = 1e-7
    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.clip(np.asarray(y_score).ravel(), eps, 1 - eps)
    return -(y_true * np.log(y_score) + (1 - y_true) * np.log(1 - y_score))


class MoETwoPretrained:
    """
    Two-pretrained Mixture-of-Experts with a logistic regression gate.

    Given two pretrained experts (A and B), learn a gate g(x) using LogisticRegression
    on a validation split, where the training label is which expert has lower per-sample loss.
    The final prediction is:
        y_hat(x) = (1 - g(x)) * s_a(x) + g(x) * s_b(x)

    Parameters
    ----------
    expert_a, expert_b : estimators
        Two pretrained models (any sklearn-like estimator).
    score_fn : callable or None
        Maps (model, X, **score_kwargs) -> score array s(x). Default: binary P(y=1).
    loss_fn : callable or None
        Maps (y_true, y_score) -> per-sample loss array. Default: binary log loss.
    gate : LogisticRegression or None
        If provided, used as the gating model; otherwise a new LogisticRegression() is created.
    use_features : {"X", "Z"}, default="Z"
        Which features to feed the gate:
          - "Z": use the provided Z_val / Z for gating (recommended: hand-picked or projected features).
          - "X": use the same X given to experts (requires numeric feature matrix).

    Notes
    -----
    - This class assumes experts are already trained. If you want in-class training, add an auto_fit flag.
    - For regression or survival tasks, pass appropriate `score_fn` and `loss_fn`.
      Example: regression loss_fn can be (y_true - y_score)**2; survival can use a time-dependent loss.
    """

    def __init__(
        self,
        expert_a,
        expert_b,
        *,
        score_fn: Optional[Callable[..., np.ndarray]] = None,
        loss_fn: Optional[Callable[..., np.ndarray]] = None,
        gate: Optional[LogisticRegression] = None,
        use_features: str = "Z",
    ):
        self.expert_a = expert_a
        self.expert_b = expert_b
        self.score_fn = score_fn or (lambda m, X, **kw: predict_prob(m, X))
        self.loss_fn = loss_fn or _default_binary_loss
        self.gate = gate or LogisticRegression(max_iter=1000)
        self.use_features = use_features

        self.is_fitted_ = False

    def fit(
        self,
        X_val,
        y_val,
        *,
        Z_val: Optional[np.ndarray] = None,
        **score_kwargs: Any,
    ):
        """
        Fit the logistic gate using a validation split.

        Parameters
        ----------
        X_val : array-like or DataFrame
            Features passed to experts to compute scores.
        y_val : array-like
            Ground-truth labels (binary/regression/survival target; depends on loss_fn).
        Z_val : array-like, optional
            Gating features. If None and use_features="Z", raises error.
            If use_features="X", the gate uses X_val directly.
        **score_kwargs :
            Extra arguments forwarded to score_fn (e.g., t_star for survival horizon).
        """
        # 1) Compute expert scores on the validation set
        s_a = np.asarray(self.score_fn(self.expert_a, X_val, **score_kwargs)).ravel()
        s_b = np.asarray(self.score_fn(self.expert_b, X_val, **score_kwargs)).ravel()

        # 2) Compute per-sample losses and gate labels (hard assignment)
        loss_a = np.asarray(self.loss_fn(y_val, s_a)).ravel()
        loss_b = np.asarray(self.loss_fn(y_val, s_b)).ravel()

        # label 1 if expert_b is better (lower loss), else 0
        y_gate = (loss_b < loss_a).astype(int)

        # 3) Choose features for the gate
        if self.use_features.upper() == "Z":
            if Z_val is None:
                raise ValueError("Z_val must be provided when use_features='Z'.")
            Z = np.asarray(Z_val)
        elif self.use_features.upper() == "X":
            Z = np.asarray(X_val)
        else:
            raise ValueError("use_features must be 'Z' or 'X'.")

        # 4) Fit logistic regression gate
        self.gate.fit(Z, y_gate)
        self.is_fitted_ = True
        return self

    def predict_proba(
        self,
        X,
        *,
        Z: Optional[np.ndarray] = None,
        **score_kwargs: Any,
    ) -> np.ndarray:
        """
        Predict the mixed score:
            (1 - g(x)) * s_a(x) + g(x) * s_b(x)

        Parameters
        ----------
        X : array-like or DataFrame
            Features for the experts.
        Z : array-like, optional
            Gating features. Required if use_features='Z'.
        **score_kwargs :
            Forwarded to score_fn.

        Returns
        -------
        y_score : np.ndarray, shape (n_samples,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Gate not fitted. Call .fit(...) first.")

        s_a = np.asarray(self.score_fn(self.expert_a, X, **score_kwargs)).ravel()
        s_b = np.asarray(self.score_fn(self.expert_b, X, **score_kwargs)).ravel()

        if self.use_features.upper() == "Z":
            if Z is None:
                raise ValueError("Z must be provided when use_features='Z'.")
            G = np.asarray(Z)
        else:
            G = np.asarray(X)

        g = self.gate.predict_proba(G)[:, 1]
        return (1.0 - g) * s_a + g * s_b
