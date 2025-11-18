from typing import Optional, Callable, Any, Dict
import numpy as np

try:
    from sklearn.base import clone
except Exception:
    clone = None  # clone is optional; if unavailable we’ll reuse the estimators

from utils.common import predict_prob


class MixtureTwoPretrained:
    """
    Two-pretrained Mixture:
      Train (or accept) two base models — a performance-based model and a fairness model — and
      combine their task-specific scores with a global weight λ:

          y_hat = (1 - λ) * s_perf + λ * s_fair

    Parameters
    ----------
    perf_model : estimator
        Performance-focused estimator (unfitted or already fitted).
    fair_model : estimator
        Fairness-focused estimator (unfitted or already fitted).
    lam : float, default=0.5
        Global mixture weight λ in [0, 1].
    score_fn : callable or None
        Function mapping (model, X, **score_kwargs) -> 1D score array.
        Defaults to binary probability via `predict_prob`.
    auto_fit : bool, default=True
        If True, `fit` will train copies of `perf_model` and `fair_model` on (X, y).
        If False, models are assumed already trained and will be used as-is.
    clone_on_fit : bool, default=True
        If True, clone the given estimators before fitting to avoid in-place modification.

    Notes
    -----
    - This class *does not* learn λ; it just applies a fixed global trade-off.
      (If you later want λ selection, we can add lam='auto' with a validation objective.)
    - `score_fn` lets you reuse the same combiner for binary/regression/survival tasks.
      For binary:  score_fn=lambda m, X: predict_prob(m, X)
      For survival (Cox risk with lifelines): score_fn=cox_risk_lifelines
    """

    def __init__(
        self,
        perf_model,
        fair_model,
        lam: float = 0.5,
        score_fn: Optional[Callable[..., np.ndarray]] = None,
        auto_fit: bool = True,
        clone_on_fit: bool = True,
    ):
        self.perf_model = perf_model
        self.fair_model = fair_model
        self.lam = float(lam)
        self.score_fn = score_fn or (lambda m, X, **kw: predict_prob(m, X))
        self.auto_fit = bool(auto_fit)
        self.clone_on_fit = bool(clone_on_fit)

        self._perf_trained = None
        self._fair_trained = None
        self.is_fitted_ = False

    def fit(
        self,
        X,
        y=None,
        *,
        fit_kwargs_perf: Optional[Dict[str, Any]] = None,
        fit_kwargs_fair: Optional[Dict[str, Any]] = None,
        **score_kwargs,
    ):
        """
        Fit the two base models if `auto_fit=True`, otherwise mark as ready.

        Parameters
        ----------
        X : array-like or dataframe
        y : array-like, optional
            Required if `auto_fit=True`.
        fit_kwargs_perf : dict, optional
            Extra kwargs for perf_model.fit (e.g., sample_weight=...).
        fit_kwargs_fair : dict, optional
            Extra kwargs for fair_model.fit.
        **score_kwargs :
            Reserved for score_fn if you later need to compute scores during fit
            (not used here; scoring is done at predict time).
        """
        fit_kwargs_perf = fit_kwargs_perf or {}
        fit_kwargs_fair = fit_kwargs_fair or {}

        if self.auto_fit:
            if y is None:
                raise ValueError("y must be provided when auto_fit=True.")

            # clone to avoid mutating user-provided estimators
            if self.clone_on_fit and clone is not None:
                perf = clone(self.perf_model)
                fair = clone(self.fair_model)
            else:
                perf = self.perf_model
                fair = self.fair_model

            # fit
            perf.fit(X, y, **fit_kwargs_perf)
            fair.fit(X, y, **fit_kwargs_fair)

            self._perf_trained = perf
            self._fair_trained = fair
        else:
            # Assume provided models are already trained
            self._perf_trained = self.perf_model
            self._fair_trained = self.fair_model

        self.is_fitted_ = True
        return self

    def _scores(self, X, **score_kwargs):
        """Compute task-aware scores for both experts."""
        s_perf = self.score_fn(self._perf_trained, X, **score_kwargs)
        s_fair = self.score_fn(self._fair_trained, X, **score_kwargs)
        s_perf = np.asarray(s_perf).ravel()
        s_fair = np.asarray(s_fair).ravel()
        if s_perf.shape != s_fair.shape:
            raise ValueError(f"Score shapes mismatch: {s_perf.shape} vs {s_fair.shape}")
        return s_perf, s_fair

    def predict_proba(self, X, **score_kwargs):
        """
        Return the mixed score:
          (1 - λ) * s_perf(X) + λ * s_fair(X)

        For binary classification with the default score_fn, this is a probability in [0,1].
        For regression/survival, it's the same scale as the chosen score_fn.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call .fit(...) first.")
        s_perf, s_fair = self._scores(X, **score_kwargs)
        return (1.0 - self.lam) * s_perf + self.lam * s_fair
