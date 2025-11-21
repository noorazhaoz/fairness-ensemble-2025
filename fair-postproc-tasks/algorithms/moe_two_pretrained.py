from typing import Optional, Callable, Any, Dict
import numpy as np
from sklearn.linear_model import LogisticRegression

from utils.gating import LinearGating   # kept only for backward compatibility
from utils.common import predict_prob


# ------------------------------------------------------------
# Default per-instance binary log-loss
# ------------------------------------------------------------
def _default_binary_loss(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """
    Per-instance log-loss for binary classification.
    y_true in {0,1}, y_score in [0,1].
    Returns an array of shape (n_samples,).
    """
    eps = 1e-7
    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.clip(np.asarray(y_score).ravel(), eps, 1 - eps)
    return -(y_true * np.log(y_score) + (1 - y_true) * np.log(1 - y_score))


# ------------------------------------------------------------
# Two-pretrained Mixture-of-Experts
# ------------------------------------------------------------
class MoETwoPretrained:
    """
    Two-pretrained Mixture-of-Experts (MoE) with a logistic regression gate.

    We assume two fixed expert models:
        Expert A: performance-oriented model (pretrained)
        Expert B: fairness-oriented model (pretrained)

    The gate g(x) is trained on a validation split.  
    Supervision for g is determined by per-instance loss comparison:

        y_gate[i] = 1  if  loss_B[i] < loss_A[i]
                     0  otherwise

    During prediction, the final mixture score is:

        y_hat(x) = (1 - g(x)) * s_A(x) + g(x) * s_B(x)

    where s_A(x) and s_B(x) are expert score functions (task-aware).

    Task Adaptation
    ---------------
    This class is task-agnostic:
        • Binary classification: default score_fn=predict_prob
        • Regression:     score_fn = m.predict,  loss_fn = (y - s)^2
        • Survival:       score_fn = risk or survival prob; custom loss_fn

    Parameters
    ----------
    expert_a : estimator
        Pretrained performance expert.

    expert_b : estimator
        Pretrained fairness expert.

    score_fn : callable, optional
        (model, X, **kw) -> score array. Default: binary P(y=1).

    loss_fn : callable, optional
        (y_true, y_score) -> per-sample loss array.
        Default: per-instance binary log-loss.

    gate : LogisticRegression or None
        If provided, used as gating model. Otherwise a new LogisticRegression(max_iter=1000) is created.

    use_features : {"Z", "X"}, default="Z"
        Feature source for the gate:
            "Z" → use the provided Z_val / Z (recommended: handcrafted or projected features)
            "X" → use the original feature matrix (must be numeric)
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

        self.use_features = use_features.upper()
        if self.use_features not in {"Z", "X"}:
            raise ValueError("use_features must be 'Z' or 'X'.")

        self.is_fitted_ = False


    # ------------------------------------------------------------
    # Fitting the gate (experts are fixed)
    # ------------------------------------------------------------
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
            Feature matrix passed to both experts to compute their scores.

        y_val : array-like
            Validation ground truth. Should match the semantics of loss_fn.

        Z_val : array-like, optional
            Gate feature matrix. Required if use_features="Z".

        **score_kwargs :
            Extra keyword arguments forwarded to score_fn (e.g., survival horizon).
        """

        # 1) Expert scores on validation data
        s_a = np.asarray(self.score_fn(self.expert_a, X_val, **score_kwargs)).ravel()
        s_b = np.asarray(self.score_fn(self.expert_b, X_val, **score_kwargs)).ravel()

        # 2) Per-sample losses
        loss_a = np.asarray(self.loss_fn(y_val, s_a)).ravel()
        loss_b = np.asarray(self.loss_fn(y_val, s_b)).ravel()

        # Gate label: 1 if fairness expert is better
        y_gate = (loss_b < loss_a).astype(int)

        # 3) Gate features
        if self.use_features == "Z":
            if Z_val is None:
                raise ValueError("Z_val must be provided when use_features='Z'.")
            G = np.asarray(Z_val)
        else:  # "X"
            G = np.asarray(X_val)

        # 4) Fit logistic regression gate
        self.gate.fit(G, y_gate)
        self.is_fitted_ = True
        return self


    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------
    def predict_proba(
        self,
        X,
        *,
        Z: Optional[np.ndarray] = None,
        **score_kwargs: Any,
    ) -> np.ndarray:
        """
        Compute the mixture score:
            (1 - g(x)) * s_A(x) + g(x) * s_B(x)

        Parameters
        ----------
        X : array-like
            Input features for experts.

        Z : array-like or None
            Gating features if use_features="Z". Required in that case.

        Returns
        -------
        score_mix : np.ndarray
            Mixed score for each instance.
        """
        if not self.is_fitted_:
            raise RuntimeError("Gate not fitted. Call .fit(...) first.")

        # Expert scores
        s_a = np.asarray(self.score_fn(self.expert_a, X, **score_kwargs)).ravel()
        s_b = np.asarray(self.score_fn(self.expert_b, X, **score_kwargs)).ravel()

        # Gate features
        if self.use_features == "Z":
            if Z is None:
                raise ValueError("Z must be provided when use_features='Z'.")
            G = np.asarray(Z)
        else:
            G = np.asarray(X)

        # Gate output probability
        g = self.gate.predict_proba(G)[:, 1]

        # Final mixture
        return (1.0 - g) * s_a + g * s_b
