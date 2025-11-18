import numpy as np

def predict_prob(model, X):
    """Binary classification: return P(y=1)."""
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:,1] if getattr(p, 'ndim', 1)==2 and p.shape[1]==2 else p.ravel()
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1.0/(1.0 + np.exp(-z))
    return getattr(model, "predict")(X).astype(float).ravel()

def predict_regression_score(model, X, normalize=False, ref=None):
    y = getattr(model, "predict")(X).astype(float).ravel()
    if normalize:
        mu, sd = (np.mean(ref), np.std(ref)+1e-8) if ref is not None else (np.mean(y), np.std(y)+1e-8)
        y = (y - mu) / sd
    return y

def _has(package):
    try:
        __import__(package)
        return True
    except Exception:
        return False

def cox_risk_lifelines(model, X):
    if not _has("lifelines"):
        raise ImportError("lifelines not installed; install lifelines to use cox_risk_lifelines")
    ph = model.predict_partial_hazard(X)
    return np.asarray(ph).ravel()

def cox_survprob_at_t_lifelines(model, X, t_star):
    if not _has("lifelines"):
        raise ImportError("lifelines not installed; install lifelines to use cox_survprob_at_t_lifelines")
    sfs = model.predict_survival_function(X)
    if t_star in sfs.index:
        s_t = sfs.loc[t_star].values
    else:
        idx = (np.abs(sfs.index.values - t_star)).argmin()
        s_t = sfs.iloc[idx].values
    return 1.0 - np.asarray(s_t).ravel()

def predict_score(model, X, task="binary", **kwargs):
    if task == "binary":
        return predict_prob(model, X)
    elif task in ("reg","regression"):
        return predict_regression_score(model, X, normalize=kwargs.get("normalize", False), ref=kwargs.get("ref", None))
    elif task == "surv_risk":
        return cox_risk_lifelines(model, X)
    elif task in ("surv_horiz","survival_horizon"):
        t_star = kwargs.get("t_star", None)
        assert t_star is not None, "t_star (horizon) is required for surv_horiz"
        return cox_survprob_at_t_lifelines(model, X, t_star=t_star)
    else:
        raise ValueError(f"Unknown task: {task}")
