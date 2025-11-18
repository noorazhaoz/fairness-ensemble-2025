from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

def classification_metrics(y_true, y_prob, thresh=0.5):
    y_pred = (y_prob >= thresh).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob, eps=1e-7),
    }
