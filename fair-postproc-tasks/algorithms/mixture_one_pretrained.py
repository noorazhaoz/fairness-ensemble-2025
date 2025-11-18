from typing import Optional, Callable, Any, Dict
import numpy as np
from sklearn.metrics import log_loss
from utils.common import predict_prob

# Default choices for binary tasks
def _default_binary_score(model, X, **_):
    return predict_prob(model, X)

def _default_binary_loss(y_true, y_score):
    """Aggregate log-loss (lower is better)."""
    return log_loss(np.asarray(y_true).ravel().astype(int),
                    np.clip(np.asarray(y_score).ravel(), 1e-7, 1-1e-7),
                    eps=1e-7)

def _default_binary_fairness(y_score, sens, **_):
    """Demographic parity difference as the fairness penalty."""
    from metrics.fairness import demographic_parity_diff
    return demographic_parity_diff(y_score, sens)


class MixtureOnePretrained:
    """
    One-pretrained Mixture (task-aware).

    - perf_model: PRETRAINED (fixed).
    - fair_model: TRAINED inside this class (during fit).
    - λ: global mixture weight; if 'auto', selected on validation by minimizing:

          objective(λ) = loss_fn(y_val, (1-λ)*s_perf + λ*s_fair) + μ * fairness_penalty((1-λ)*s_perf + λ*s_fair, sens_val)

    Scores s_* are produced by `score_fn`, which makes the class work for
    binary / regression / survival.

    Parameters
    ----------
    perf_model : estimator (already trained)
        Performance-focused model; NOT refit here.
    fair_model : estimator (unfitted)
        Fairness-focused model; will be fit on (X, y) in .fit().
        (For binary: LogisticRegression; regression: LinearRegression; survival: Cox, etc.)
    lam : float or 'auto', default='auto'
        If float in [0,1], use that weight. If 'auto', select λ on validation.
    mu : float, default=0.0
        Weight of fairness penalty in the validation objective (when lam='auto').
    lam_grid : array-like or None
        Candidate λ values when lam='auto'. Default: np.linspace(0,1,21).
    score_fn : callable or None
        (model, X, **score_kwargs) -> 1D score array. Default: binary P(y=1).
    loss_fn : callable or None
        (y_true, y_score, **score_kwargs) -> scalar loss (lower is better).
        Default: binary log-loss.
    fairness_penalty : callable or None
        (y_score, sens, **score_kwargs) -> non-negative scalar. Default: DP diff (binary).
    auto_fit_fair : bool, default=True
        If True, fit the fairness model on (X, y) during .fit().
    fit_kwargs_fair : dict or None
        Extra kwargs passed to fair_model.fit.
    """

    def __init__(
        self,
        perf_model,
        fair_model,
        lam: Any = "auto",
        mu: float = 0.0,
        lam_grid: Optional[np.ndarray] = None,
        score_fn: Optional[Callable[..., np.ndarray]] = None,
        loss_fn: Optional[Callable[..., float]] = None,
        fairness_penalty: Optional[Callable[..., float]] = None,
        *,
        auto_fit_fair: bool = True,
        fit_kwargs_fair: Optional[Dict[str, Any]] = None,
    ):
        self.perf_model = perf_model
        self.fair_model = fair_model
        self.lam = lam
        self.mu = float(mu)
        self.lam_grid = lam_grid if lam_grid is not None else np.linspace(0.0, 1.0, 21)

        # task-aware hooks
        self.score_fn = score_fn or _default_binary_score
        self.loss_fn = loss_fn or _default_binary_loss
        self.fairness_penalty = fairness_penalty or _default_binary_fairness

        self.auto_fit_fair = bool(auto_fit_fair)
        self.fit_kwargs_fair = fit_kwargs_fair or {}

        self._fair_trained = None
        self.is_fitted_ = False

    # ---------- internal helpers ----------

    def _scores(self, X, **score_kwargs):
        s_perf = np.asarray(self.score_fn(self.perf_model, X, **score_kwargs)).ravel()
        s_fair = np.asarray(self.score_fn(self._fair_trained, X, **score_kwargs)).ravel()
        if s_perf.shape != s_fair.shape:
            raise ValueError(f"Score shapes mismatch: {s_perf.shape} vs {s_fair.shape}")
        return s_perf, s_fair

    def _objective(self, y_true, y_score, sens=None, **score_kwargs):
        loss = self.loss_fn(y_true, y_score, **score_kwargs)
        penalty = 0.0 if sens is None else self.fairness_penalty(y_score, sens, **score_kwargs)
        return float(loss) + self.mu * float(penalty)

    # --------------- API ------------------

    def fit(self, X, y=None, *, X_val=None, y_val=None, sens_val=None, **score_kwargs):
        """
        Train the fairness model (if auto_fit_fair=True) and select λ if needed.

        For survival/regression:
        - Provide suitable score_fn and loss_fn/fairness_penalty.
        - y/y_val/sens_val should match your functions' expectations.

        Parameters
        ----------
        X, y : training data for fairness model
        X_val, y_val : validation data for λ selection when lam='auto'
        sens_val : sensitive attribute on validation set (for fairness penalty)
        **score_kwargs : forwarded to score_fn / loss_fn / fairness_penalty (e.g., t_star for survival horizon)
        """
        # 1) Fit fairness model (perf model is assumed pretrained)
        if self.auto_fit_fair:
            if y is None:
                raise ValueError("y must be provided to fit the fairness model.")
            self.fair_model.fit(X, y, **self.fit_kwargs_fair)
            self._fair_trained = self.fair_model
        else:
            # assume user already trained fairness model
            self._fair_trained = self.fair_model

        # 2) λ selection on validation, if requested
        if self.lam == "auto":
            if X_val is None or y_val is None:
                raise ValueError("X_val and y_val are required when lam='auto'.")
            s_perf = np.asarray(self.score_fn(self.perf_model, X_val, **score_kwargs)).ravel()
            s_fair = np.asarray(self.score_fn(self._fair_trained, X_val, **score_kwargs)).ravel()
            if s_perf.shape != s_fair.shape:
                raise ValueError(f"Score shapes mismatch on val: {s_perf.shape} vs {s_fair.shape}")

            best_obj, best_lam = np.inf, 0.0
            for lam in self.lam_grid:
                y_mix = (1.0 - lam) * s_perf + lam * s_fair
                obj = self._objective(y_val, y_mix, sens=sens_val, **score_kwargs)
                if obj < best_obj:
                    best_obj, best_lam = obj, lam
            self.lam = float(best_lam)
        else:
            # ensure float
            self.lam = float(self.lam)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X, **score_kwargs):
        """
        Return mixed score: (1 - λ) * s_perf + λ * s_fair

        - Binary: probability in [0,1] if score_fn returns probabilities.
        - Regression/Survival: task-scale score as defined by score_fn.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call .fit(...) first.")
        s_perf, s_fair = self._scores(X, **score_kwargs)
        return (1.0 - float(self.lam)) * s_perf + float(self.lam) * s_fair
