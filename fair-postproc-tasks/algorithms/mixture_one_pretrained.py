from typing import Optional, Callable, Any, Dict
import numpy as np
from sklearn.metrics import log_loss
from utils.common import predict_prob

# ----------------------------------------------------------------------
# Default hooks for binary classification
# ----------------------------------------------------------------------
def _default_binary_score(model, X, **_):
    """
    Default scoring function for binary tasks:
    returns P(y=1 | x) via utils.common.predict_prob.
    """
    return predict_prob(model, X)


def _default_binary_loss(y_true, y_score):
    """
    Default aggregate loss for binary tasks: log-loss (cross-entropy).
    Lower is better.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.clip(np.asarray(y_score).ravel(), 1e-7, 1 - 1e-7)
    return log_loss(y_true, y_score, eps=1e-7)


def _default_binary_fairness(y_score, sens, **_):
    """
    Default fairness penalty for binary tasks:
    Demographic parity difference computed on score > threshold.
    """
    from metrics.fairness import demographic_parity_diff
    return demographic_parity_diff(y_score, sens)


class MixtureOnePretrained:
    """
    One-pretrained Mixture (task-aware, with optional λ selection).

    This class implements a model-agnostic **post-processing** scheme with:
      - a PRETRAINED performance model `perf_model`, kept fixed; and
      - a fairness-oriented model `fair_model`, trained inside this class.

    For an input x, each model produces a task-specific score:
        s_perf(x) = score_fn(perf_model, x)
        s_fair(x) = score_fn(fair_model, x)

    We then combine them with a global mixture weight λ:

        s_mix(x) = (1 - λ) * s_perf(x) + λ * s_fair(x).

    When `lam='auto'`, λ is selected on a validation set by minimizing:

        objective(λ)
          = loss_fn(y_val, s_mix_λ)
            + μ * fairness_penalty(s_mix_λ, sens_val),

        where s_mix_λ = (1 - λ) * s_perf + λ * s_fair.

    The hooks `score_fn`, `loss_fn`, and `fairness_penalty` make this class
    usable for binary classification, regression, or survival tasks.

    Typical usage patterns (in our experiments)
    ------------------------------------------
    - Binary classification (Adult, COMPAS, German Credit):
        * perf_model: RF or MLP on all features.
        * fair_model: simple LR on a restricted feature subset
                      (e.g. 'education-num' or 'bmi').
        * score_fn:  P(y=1 | x)
        * loss_fn:   log-loss
        * fairness_penalty: Demographic parity or Equalized Odds (via custom hook).

    - Regression:
        * score_fn:  model.predict(X)
        * loss_fn:   MSE or MAE
        * fairness_penalty: e.g. max statistical parity in prediction.

    - Survival:
        * score_fn:  risk score or survival probability at a fixed time grid.
        * loss_fn / fairness_penalty adapted accordingly.

    Parameters
    ----------
    perf_model : estimator (already trained)
        Performance-focused model; NOT refit here. In our code we typically
        train perf_model externally, then pass it into this class.

    fair_model : estimator (unfitted)
        Fairness-focused model; will be fit on (X, y) inside `.fit()` if
        `auto_fit_fair=True`. In practice this is often a simple model
        (e.g., LogisticRegression on a single feature or a small subset).

    lam : float or 'auto', default='auto'
        Global mixture weight in [0, 1] OR the string 'auto'.
        - If a float, that λ is used as-is.
        - If 'auto', λ is chosen from `lam_grid` by minimizing the
          validation objective on (X_val, y_val, sens_val).

    mu : float, default=0.0
        Weight of the fairness penalty in the validation objective:
            objective = loss_fn + mu * fairness_penalty
        (only used when lam='auto').

    lam_grid : array-like or None
        Grid of candidate λ values when `lam='auto'`.
        Default: np.linspace(0, 1, 21).

    score_fn : callable or None
        Scoring function:
            score_fn(model, X, **score_kwargs) -> 1D score array.
        Default: `_default_binary_score` (probabilities for binary tasks).

    loss_fn : callable or None
        Task loss:
            loss_fn(y_true, y_score, **score_kwargs) -> scalar (lower is better).
        Default: `_default_binary_loss` (log-loss for binary tasks).

    fairness_penalty : callable or None
        Fairness penalty:
            fairness_penalty(y_score, sens, **score_kwargs) -> non-negative scalar.
        Default: `_default_binary_fairness` (Demographic Parity difference).

    auto_fit_fair : bool, default=True
        If True, call `fair_model.fit(X, y, **fit_kwargs_fair)` inside `.fit()`.
        If False, we assume `fair_model` is already trained and simply reuse it.

    fit_kwargs_fair : dict or None
        Extra keyword arguments passed to `fair_model.fit`.

    Notes
    -----
    - This class **never** retrains `perf_model`.
    - The fairness model `fair_model` can be any estimator (LR, LinearRegression,
      CoxPH, etc.) as long as `score_fn` knows how to obtain a 1D score.
    - The design choices (which feature subset is used for fair_model, what
      fairness_penalty is used, etc.) are dataset/task-specific and are set
      outside this class.
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

        # λ and fairness weight μ
        self.lam = lam
        self.mu = float(mu)
        self.lam_grid = (
            lam_grid if lam_grid is not None else np.linspace(0.0, 1.0, 21)
        )

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
        """
        Compute scores from perf_model (fixed) and trained fairness model
        using the user-provided `score_fn`.
        """
        s_perf = np.asarray(self.score_fn(self.perf_model, X, **score_kwargs)).ravel()
        s_fair = np.asarray(self.score_fn(self._fair_trained, X, **score_kwargs)).ravel()
        if s_perf.shape != s_fair.shape:
            raise ValueError(f"Score shapes mismatch: {s_perf.shape} vs {s_fair.shape}")
        return s_perf, s_fair

    def _objective(self, y_true, y_score, sens=None, **score_kwargs):
        """
        Combined objective:
            loss_fn(y_true, y_score) + mu * fairness_penalty(y_score, sens).
        """
        loss = self.loss_fn(y_true, y_score, **score_kwargs)
        penalty = 0.0 if sens is None else self.fairness_penalty(
            y_score, sens, **score_kwargs
        )
        return float(loss) + self.mu * float(penalty)

    # --------------- public API ------------------

    def fit(self, X, y=None, *, X_val=None, y_val=None, sens_val=None, **score_kwargs):
        """
        Train the fairness model (if `auto_fit_fair=True`) and optionally
        select the mixture weight λ on a validation set.

        For non-binary tasks, the user should supply appropriate
        `score_fn`, `loss_fn`, and `fairness_penalty`, and ensure that
        y / y_val / sens_val follow those functions' expectations.

        Parameters
        ----------
        X, y : training data for fairness model.
            - X : feature matrix
            - y : labels or targets (binary/regression/survival, etc.)

        X_val, y_val : validation data for λ selection when lam='auto'.
        sens_val : sensitive attribute on the validation set (for fairness penalty).

        **score_kwargs :
            Additional keyword arguments forwarded to score_fn, loss_fn,
            and fairness_penalty (e.g. survival horizons).
        """
        # 1) Fit the fairness model (perf_model is assumed pretrained)
        if self.auto_fit_fair:
            if y is None:
                raise ValueError("y must be provided to fit the fairness model.")
            self.fair_model.fit(X, y, **self.fit_kwargs_fair)
            self._fair_trained = self.fair_model
        else:
            # Assume user already trained fairness model
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
            # Ensure λ is a float in [0,1] when provided directly
            self.lam = float(self.lam)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X, **score_kwargs):
        """
        Return mixed score:
            s_mix(x) = (1 - λ) * s_perf(x) + λ * s_fair(x).

        - For binary tasks with the default `score_fn`, this is a probability
          in [0, 1].
        - For regression or survival tasks, s_mix is on the same scale as the
          scores returned by `score_fn` (e.g. predicted value or risk score).
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call .fit(...) first.")
        s_perf, s_fair = self._scores(X, **score_kwargs)
        return (1.0 - float(self.lam)) * s_perf + float(self.lam) * s_fair
