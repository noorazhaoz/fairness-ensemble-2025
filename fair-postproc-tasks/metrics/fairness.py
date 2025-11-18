import numpy as np


# ----------------------------- #
# Binary classification metrics #
# ----------------------------- #

def demographic_parity_diff(y_pred, sens, threshold: float = 0.5):
    """
    Demographic parity difference for binary classification.

    Parameters
    ----------
    y_pred : array-like, shape (n_samples,)
        Predicted scores or probabilities; will be thresholded at `threshold`.
    sens : array-like, shape (n_samples,)
        Sensitive attribute; binary (0/1).
    threshold : float, default=0.5
        Threshold to binarize predictions.

    Returns
    -------
    float
        |P(ŷ=1|s=1) - P(ŷ=1|s=0)|.
        Returns np.nan if either group has no samples.
    """
    y_pred = np.asarray(y_pred).ravel()
    sens = np.asarray(sens).ravel().astype(int)

    if (sens == 1).sum() == 0 or (sens == 0).sum() == 0:
        return np.nan

    y_bin = (y_pred >= threshold).astype(int)
    p1 = y_bin[sens == 1].mean()
    p0 = y_bin[sens == 0].mean()
    return float(abs(p1 - p0))


def equalized_odds_diff(y_true, y_pred, sens, threshold: float = 0.5):
    """
    Equalized odds difference for binary classification.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        True binary labels (0/1).
    y_pred : array-like, shape (n_samples,)
        Predicted scores or probabilities; thresholded at `threshold`.
    sens : array-like, shape (n_samples,)
        Sensitive attribute; binary (0/1).
    threshold : float, default=0.5

    Returns
    -------
    float
        max( |P(ŷ=1|y=1,s=1) - P(ŷ=1|y=1,s=0)|,
             |P(ŷ=1|y=0,s=1) - P(ŷ=1|y=0,s=0)| )
        Returns np.nan if any needed subgroup is empty.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel()
    sens = np.asarray(sens).ravel().astype(int)

    y_bin = (y_pred >= threshold).astype(int)
    diffs = []

    for yv in (0, 1):
        mask = (y_true == yv)
        if mask.sum() == 0:
            continue
        mask1 = mask & (sens == 1)
        mask0 = mask & (sens == 0)
        if mask1.sum() == 0 or mask0.sum() == 0:
            continue
        p1 = (y_bin[mask1] == 1).mean()
        p0 = (y_bin[mask0] == 1).mean()
        diffs.append(abs(p1 - p0))

    return float(np.max(diffs)) if len(diffs) else np.nan


# ------------------ #
# Regression metric  #
# ------------------ #

def _regression_checks(group_a, group_b, y_pred):
    """
    Minimal input checks to mimic the behavior you referenced.
    Ensures 1-D arrays, numeric predictions, and binary group flags.
    """
    group_a = np.asarray(group_a).ravel().astype(int)
    group_b = np.asarray(group_b).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(float)

    if not (set(np.unique(group_a)) <= {0, 1} and set(np.unique(group_b)) <= {0, 1}):
        raise ValueError("group_a and group_b must be binary {0,1} vectors.")

    if len(group_a) != len(group_b) or len(group_a) != len(y_pred):
        raise ValueError("group_a, group_b, and y_pred must have the same length.")

    if group_a.sum() == 0 or group_b.sum() == 0:
        # One of the groups has no samples: metric undefined.
        return group_a, group_b, y_pred, True
    return group_a, group_b, y_pred, False


def statistical_parity_auc(group_a, group_b, y_pred, num_thresholds: int = 150):
    """
    Statistical parity (AUC) for regression-style predictions.

    This computes the area under the curve of absolute demographic
    parity difference across a set of thresholds. At each threshold τ,
    we define "pass" as y_pred >= τ (like a continuous-to-binary cut),
    then measure |pass_rate_a - pass_rate_b|, and average over τ.

    Interpretation
    --------------
    A value of 0 is desired; values below ~0.075 are often considered acceptable.

    Parameters
    ----------
    group_a : array-like (binary)
    group_b : array-like (binary)
    y_pred : array-like (float)
    num_thresholds : int, default=150
        Number of quantile thresholds in [1, 0] to evaluate.

    Returns
    -------
    float
        Statistical parity AUC. Returns np.nan if any group has zero samples.
    """
    group_a, group_b, y_pred, degenerate = _regression_checks(group_a, group_b, y_pred)
    if degenerate:
        return np.nan

    # thresholds: quantiles from 1 -> 0 (high to low)
    q = np.linspace(1.0, 0.0, num_thresholds)
    pass_value = np.quantile(y_pred, q)

    # For each threshold, binarize predictions
    y_bin = y_pred.reshape(-1, 1) >= pass_value.reshape(1, -1)

    # Pass rates by group
    pass_a = y_bin[group_a == 1].sum(axis=0) / group_a.sum()
    pass_b = y_bin[group_b == 1].sum(axis=0) / group_b.sum()

    di_arr = np.abs(pass_a - pass_b)  # disparity at each threshold

    # Average over thresholds (equally spaced in quantile domain)
    return float(np.mean(di_arr))


# ---------------- #
# Survival metric  #
# ---------------- #

def survival_group_fairness(prediction, sens, concentration: float = 1.0, num_classes: int = 2):
    """
    Group fairness for survival risk-like predictions with Dirichlet smoothing.

    Parameters
    ----------
    prediction : array-like, shape (n_samples,)
        Risk-like scores (e.g., Cox partial hazard) or event probability at a horizon.
    sens : array-like, shape (n_samples,)
        Sensitive attribute (can be multi-group; not limited to binary).
    concentration : float, default=1.0
        Total mass for symmetric Dirichlet prior.
    num_classes : int, default=2
        Number of pseudo-classes for smoothing (kept from your original formula).

    Returns
    -------
    float
        max_g | smoothed_mean_g - global_mean |
    """
    pred = np.asarray(prediction).ravel().astype(float)
    sens = np.asarray(sens).ravel()

    unique = np.unique(sens)
    if len(unique) == 0:
        return np.nan

    # Global mean of the risk/probability
    global_mean = float(np.mean(pred))

    # Dirichlet smoothing parameters
    alpha = concentration / float(num_classes)

    group_sum = {g: 0.0 for g in unique}
    group_cnt = {g: 0.0 for g in unique}

    for p, s in zip(pred, sens):
        group_sum[s] += float(p)
        group_cnt[s] += 1.0

    smoothed_means = []
    for g in unique:
        sm = (group_sum[g] + alpha) / (group_cnt[g] + concentration)
        smoothed_means.append(sm)

    smoothed_means = np.asarray(smoothed_means, dtype=float)
    return float(np.max(np.abs(smoothed_means - global_mean)))
