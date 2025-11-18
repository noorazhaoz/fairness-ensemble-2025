from typing import Optional, Callable, Any
import numpy as np
from sklearn.linear_model import LogisticRegression

from utils.common import predict_prob  # 默认二分类得分

def _default_binary_loss(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """逐样本二分类 log-loss（越小越好），输出 shape=(n_samples,)."""
    eps = 1e-7
    y_true = np.asarray(y_true).ravel().astype(int)
    y_score = np.clip(np.asarray(y_score).ravel(), eps, 1 - eps)
    return -(y_true * np.log(y_score) + (1 - y_true) * np.log(1 - y_score))


class MoEOnePretrained:
    """
    One-pretrained MoE with a logistic gate.

    设 A=performance 模型、B=fairness 模型（已预训练且固定）。
    训练流程：
      1) 在 (X_train, y_train) 上训练 A（performance）。
      2) 在验证集上比较逐样本损失，得到 gate 监督：y_gate = 1[loss_B < loss_A]
      3) 用 LogisticRegression 在 Z_val（或 X_val）上拟合 g(x)=P(y_gate=1|·)
      4) 预测时：ŷ(x) = (1 - g(x)) * s_A(x) + g(x) * s_B(x)

    参数
    ----
    perf_model : 未训练的 performance 模型（会在 fit 中训练）
    fair_model : 已训练的 fairness 模型（固定不变）
    gate : LogisticRegression 或 None（默认 LogisticRegression(max_iter=1000)）
    score_fn : (model, X, **kw) -> 1D 分数，默认二分类概率 predict_prob
    loss_fn  : (y_true, y_score) -> 逐样本损失数组，默认二分类逐样本 log-loss
    use_features : {"Z", "X"}，gate 使用的特征来源；"Z" 表示传入的 x1d/手工特征

    适配任务
    -------
    - Binary: 默认即可（score_fn=predict_prob, loss_fn=逐样本log-loss）
    - Regression: score_fn=lambda m,X: m.predict(X); loss_fn=lambda y,s:(y-s)**2
    - Survival: score_fn=风险或时点事件概率；loss_fn 可换成合适的生存 surrogate
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
        self.fair_model = fair_model  # 预训练、固定
        self.gate = gate or LogisticRegression(max_iter=1000)
        self.score_fn = score_fn or (lambda m, X, **kw: predict_prob(m, X))
        self.loss_fn = loss_fn or _default_binary_loss
        self.use_features = use_features.upper()
        if self.use_features not in {"Z", "X"}:
            raise ValueError("use_features must be 'Z' or 'X'.")

        self._perf_trained = None
        self.is_fitted_ = False

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
        训练 performance 模型和 gate（fairness 模型保持固定）。

        参数
        ----
        X_train, y_train : 用于训练 performance 模型
        X_val, y_val     : gate 监督信号的验证集（比较两专家逐样本损失）
        Z_val            : gate 的特征（若 use_features='Z' 则必需）
        **score_kwargs   : 透传给 score_fn（如 survival 的 t_star）
        """
        # 1) 训练 performance 模型
        self.perf_model.fit(X_train, y_train)
        self._perf_trained = self.perf_model

        # 2) 在验证集上计算两专家分数
        s_perf = np.asarray(self.score_fn(self._perf_trained, X_val, **score_kwargs)).ravel()
        s_fair = np.asarray(self.score_fn(self.fair_model,     X_val, **score_kwargs)).ravel()

        # 3) 逐样本损失与 gate 标签（1 表示选择 fairness 专家 B）
        loss_a = np.asarray(self.loss_fn(y_val, s_perf)).ravel()
        loss_b = np.asarray(self.loss_fn(y_val, s_fair)).ravel()
        y_gate = (loss_b < loss_a).astype(int)

        # 4) gate 特征
        if self.use_features == "Z":
            if Z_val is None:
                raise ValueError("Z_val must be provided when use_features='Z'.")
            G = np.asarray(Z_val)
        else:  # "X"
            G = np.asarray(X_val)

        # 5) 拟合 logistic gate
        self.gate.fit(G, y_gate)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X, *, Z: Optional[np.ndarray] = None, **score_kwargs: Any) -> np.ndarray:
        """
        返回混合分数： (1 - g(x)) * s_perf(x) + g(x) * s_fair(x)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model not fitted. Call .fit(...) first.")

        s_perf = np.asarray(self.score_fn(self._perf_trained, X, **score_kwargs)).ravel()
        s_fair = np.asarray(self.score_fn(self.fair_model,     X, **score_kwargs)).ravel()

        if self.use_features == "Z":
            if Z is None:
                raise ValueError("Z must be provided when use_features='Z'.")
            G = np.asarray(Z)
        else:
            G = np.asarray(X)

        g = self.gate.predict_proba(G)[:, 1]
        return (1.0 - g) * s_perf + g * s_fair
