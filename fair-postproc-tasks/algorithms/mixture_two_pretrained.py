from typing import Optional, Callable, Any, Dict
import numpy as np

try:
    from sklearn.base import clone
except Exception:
    clone = None  # clone is optional; if unavailable we’ll reuse the estimators

from utils.common import predict_prob


class MixtureTwoPretrained:
    """
    Two-pretrained Mixture (model-agnostic, post-processing).

    We assume TWO base models are already chosen:

      - A performance-focused model  f_perf
      - A simpler / more interpretable fairness-oriented model  f_fair
        (often trained on a restricted feature set, e.g. a single feature
         such as 'education-num' or 'lenstay', but this is dataset-specific
         and NOT enforced by this class).

    At prediction time, we combine their task-specific scores with a
    global mixture weight λ:

        s_mix(x) = (1 - λ) * s_perf(x) + λ * s_fair(x),

    where s_perf, s_fair are scores produced by `score_fn` (e.g., predicted
    probabilities for binary classification, regression outputs, or
    survival probabilities/risk scores).

    Parameters
    ----------
    perf_model : estimator
        Performance-focused estimator (unfitted or already fitted).
        Typically uses the full feature set.

    fair_model : estimator
        Fairness-oriented (often simpler) estimator.
        In practice this may be trained on a restricted subset of features
        (e.g., a single feature) via a ColumnTransformer / Pipeline.
        This class does not assume or enforce a particular subset; the
        user controls that when constructing `fair_model`.

    lam : float, default=0.5
        Global mixture weight λ in [0, 1].
        λ = 0   → use only the performance model.
        λ = 1   → use only the fairness-oriented model.

    score_fn : callable or None
        Function mapping (model, X, **score_kwargs) -> 1D score array.
        This makes the class task-agnostic.

        Typical choices:
        - Binary classification:
              score_fn=lambda m, X, **kw: predict_prob(m, X)
        - Regression:
              score_fn=lambda m, X, **kw: m.predict(X)
        - Survival (e.g. risk score):
              score_fn=cox_risk_lifelines   # user-defined
          or survival probability at a fixed time grid.

        By default, we use `utils.common.predict_prob`, which returns
        P(y=1 | x) for binary classifiers.

    auto_fit : bool, default=True
        If True, `fit` will train copies of `perf_model` and `fair_model`
        on (X, y). If False, models are assumed already trained and will
        be used as-is.

    clone_on_fit : bool, default=True
        If True and sklearn.clone is available, clone the given estimators
        before fitting, to avoid modifying user-provided objects in-place.

    Notes
    -----
    - This class *does not* learn λ. It simply applies a FIXED global
      trade-off chosen by the user (e.g., λ selected on a validation set
      via a fairness–performance objective defined elsewhere).

    - The choice of `fair_model` architecture and its feature subset is
      dataset-dependent. For example:
          * Adult (binary):  fair_model = LR on 'education-num'
          * Insurance (regression): fair_model = LinearRegression on 'bmi'
          * WHAS (survival): fair_model = CoxPH on 'lenstay'

      All these are compatible as long as `score_fn` returns a 1D score.
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
        # default for binary classification: probability of class 1
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
        Fit the two base models if `auto_fit=True`, otherwise just attach
        the user-provided trained models.

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
            Reserved for score_fn if you later need to pass arguments at
            fit time (here we only score at prediction time).
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
            s_mix(x) = (1 - λ) * s_perf(x) + λ * s_fair(x).

        For binary classification with the default score_fn, this is a
        probability in [0,1].

        For regression/survival, s_mix is on the same scale as the scores
        returned by `score_fn` (e.g., predicted value or risk score).
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call .fit(...) first.")
        s_perf, s_fair = self._scores(X, **score_kwargs)
        return (1.0 - self.lam) * s_perf + self.lam * s_fair
