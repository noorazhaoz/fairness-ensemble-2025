from typing import Optional, Callable, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from utils.common import predict_prob   # Default scoring: binary probability


# ------------------------------------------------------------
# Default loss: per-instance binary log-loss
# ------------------------------------------------------------
def _default_binary_loss(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """
    Per-instance binary log-loss (lower is better).
    Returns an array of shape (n_samples,).
    """
    eps = 1e-7
    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.clip(np.asarray(y_score).ravel(), eps, 1 - eps)
    return -(y_true * np.log(y_score) + (1 - y_true) * np.log(1 - y_score))


# ------------------------------------------------------------
# One-pretrained Mixture-of-Experts (MoE)
# ------------------------------------------------------------
class MoEOnePretrained:
    """
    One-pretrained Mixture-of-Experts (MoE) with a logistic gating function.

    We have two “experts”:
        A = performance model
        B = fairness model (pretrained and fixed)

    Training procedure:
    -------------------
      1) Fit the performance model A on (X_train, y_train).
      2) On the validation set, compute per-instance losses:
              loss_A[i] = loss_fn(y_val[i], score_A[i])
              loss_B[i] = loss_fn(y_val[i], score_B[i])
      3) Gate labels (supervision):
              y_gate[i] = 1  if loss_B[i] < loss_A[i]
                        = 0  otherwise
         This means: use the fairness expert B when it performs better.
      4) Fit a logistic regression gate g(·) on either:
            - Z_val  (1D handcrafted / projected features), or
            - X_val  (full features)
      5) At prediction time:
            g(x) = P(gate=1 | x)
            final_score = (1 - g(x)) * s_A(x) + g(x) * s_B(x)

    Task Adaptation:
    ----------------
    This class is task-agnostic:
      • Binary classification:
            score_fn = predict_prob (default)
            loss_fn  = per-instance log-loss (default)
      • Regression:
            score_fn = lambda m, X: m.predict(X)
            loss_fn  = lambda y, s: (y - s)**2
      • Survival:
            score_fn = risk score or survival probability
            loss_fn  = custom survival surrogate

    Parameters
    ----------
    perf_model : estimator
        Performance model (NOT pretrained). Will be trained inside fit().

    fair_model : estimator
        Fairness model (already pretrained). Kept fixed.

    gate : LogisticRegression or None
        Gating classifier. Default: LogisticRegression(max_iter=1000).

    score_fn : callable or None
        (model, X, **kw) -> 1D array of scores.
        Default: binary probability predict_prob().

    loss_fn : callable or None
        (y_true, y_score) -> per-instance loss array.
        Default: per-instance binary log-loss.

    use_features : {"Z", "X"}
        Specifies the feature input for the gate:
            "Z" → use provided Z_val / Z
            "X" → use the original feature matrix (X_val / X)
    """

    def __init__(
        self,
        perf_model,
        fair_model,
        *,
        gate: Optional[LogisticRegression] = None,
        score_fn: Optional[Callable[..., np.ndarray]] = None,
        loss_fn: Optional[Callable[..., np.ndarray]] = None,
        use_features: str = "Z",
    ):
        self.perf_model = perf_model
        self.fair_model = fair_model   # Pretrained fairness expert
        self.gate = gate or LogisticRegression(max_iter=1000)

        self.score_fn = score_fn or (lambda m, X, **kw: predict_prob(m, X))
        self.loss_fn = loss_fn or _default_binary_loss

        self.use_features = use_features.upper()
        if self.use_features not in {"Z", "X"}:
            raise ValueError("use_features must be 'Z' or 'X'.")

        self._perf_trained = None
        self.is_fitted_ = False


    # ------------------------------------------------------------
    # Fit MoE: train performance model + gate (fair model fixed)
    # ------------------------------------------------------------
    def fit(
        self,
        X_train,
        y_train,
        *,
        X_val,
        y_val,
        Z_val: Optional[np.ndarray] = None,
        **score_kwargs: Any,
    ):
        """
        Fit the performance model and the logistic gate.

        Parameters
        ----------
        X_train, y_train : training data for the performance model.

        X_val, y_val : validation data for computing loss differences
                        and generating gate supervision.

        Z_val : optional array
            Additional handcrafted / projected features for the gate.
            Required if use_features='Z'.

        score_kwargs : optional keyword arguments forwarded to score_fn
                       (e.g., survival time horizon)
        """

        # 1) Train performance expert A
        self.perf_model.fit(X_train, y_train)
        self._perf_trained = self.perf_model

        # 2) Compute expert scores on validation data
        s_A = np.asarray(self.score_fn(self._perf_trained, X_val, **score_kwargs)).ravel()
        s_B = np.asarray(self.score_fn(self.fair_model,     X_val, **score_kwargs)).ravel()

        # 3) Compute per-instance losses
        loss_A = np.asarray(self.loss_fn(y_val, s_A)).ravel()
        loss_B = np.asarray(self.loss_fn(y_val, s_B)).ravel()

        # 4) Gate supervision: 1 means choose fairness expert
        y_gate = (loss_B < loss_A).astype(int)

        # 5) Gate features
        if self.use_features == "Z":
            if Z_val is None:
                raise ValueError("Z_val must be provided when use_features='Z'.")
            G = np.asarray(Z_val)
        else:
            G = np.asarray(X_val)

        # 6) Fit the logistic gate
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
        **score_kwargs: Any
    ) -> np.ndarray:
        """
        Mixed prediction:
            (1 - g(x)) * score_A(x) + g(x) * score_B(x)

        Parameters
        ----------
        X : array-like
            Input features for experts and for gate (if use_features='X').

        Z : array-like or None
            Gate features when use_features='Z'. Required in that case.

        Returns
        -------
        y_score : np.ndarray
            Mixed expert score for each instance.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        s_A = np.asarray(self.score_fn(self._perf_trained, X, **score_kwargs)).ravel()
        s_B = np.asarray(self.score_fn(self.fair_model,     X, **score_kwargs)).ravel()

        # Gate features
        if self.use_features == "Z":
            if Z is None:
                raise ValueError("Z must be provided when use_features='Z'.")
            G = np.asarray(Z)
        else:
            G = np.asarray(X)

        # Gate output probability
        g = self.gate.predict_proba(G)[:, 1]

        # Mixture output
        return (1.0 - g) * s_A + g * s_B
