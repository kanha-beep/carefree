from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import FEATURE_FILE, FEATURE_IMPORTANCE_FILE, MODEL_DIR, MODEL_FILE, MODEL_METRICS_FILE, REPORT_DIR
from src.pipeline.database import record_pipeline_run
from src.pipeline.logging_utils import get_pipeline_logger
from src.models.explainability import compute_permutation_importance
from src.models.pytorch_model import ReadmissionPredictor
from src.run_pipeline import main as run_pipeline
from uuid import uuid4


def _split_data(
    x: np.ndarray, y: np.ndarray, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(x))
    x = x[indices]
    y = y[indices]

    train_end = int(len(x) * train_ratio)
    val_end = train_end + int(len(x) * val_ratio)

    return (
        x[:train_end],
        x[train_end:val_end],
        x[val_end:],
        y[:train_end],
        y[train_end:val_end],
        y[val_end:],
    )


def _standardize(
    train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std[std == 0] = 1.0

    return (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.0

    total_pairs = len(positives) * len(negatives)
    wins = 0.0
    for positive_score in positives:
        wins += float((positive_score > negatives).sum())
        wins += 0.5 * float((positive_score == negatives).sum())
    return wins / total_pairs


def _find_best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = _classification_metrics(y_true, (y_score >= 0.5).astype(int))

    for threshold in np.linspace(0.1, 0.9, 17):
        candidate_predictions = (y_score >= threshold).astype(int)
        candidate_metrics = _classification_metrics(y_true, candidate_predictions)
        if candidate_metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = candidate_metrics

    return best_threshold, best_metrics


def main() -> None:
    run_id = str(uuid4())
    logger = get_pipeline_logger()
    torch.manual_seed(42)
    np.random.seed(42)

    if not FEATURE_FILE.exists():
        print("Feature file not found. Running pipeline first...")
        run_pipeline()

    df = pd.read_csv(FEATURE_FILE)

    feature_columns = [column for column in df.columns if column not in {"encounter_id", "patient_nbr", "target"}]
    x = df[feature_columns].astype(float).to_numpy()
    y = df["target"].astype(float).to_numpy().reshape(-1, 1)

    train_x, val_x, test_x, train_y, val_y, test_y = _split_data(x, y)
    train_mean = train_x.mean(axis=0, keepdims=True)
    train_std = train_x.std(axis=0, keepdims=True)
    train_std[train_std == 0] = 1.0
    train_x, val_x, test_x = _standardize(train_x, val_x, test_x)

    train_dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
    )
    val_x_tensor = torch.tensor(val_x, dtype=torch.float32)
    val_y_tensor = torch.tensor(val_y, dtype=torch.float32)
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    positive_count = float(train_y.sum())
    negative_count = float(len(train_y) - positive_count)
    pos_weight_value = negative_count / max(positive_count, 1.0)

    model = ReadmissionPredictor(input_dim=train_x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        model.eval()
        with torch.no_grad():
            val_logits = model(val_x_tensor)
            val_loss = loss_fn(val_logits, val_y_tensor).item()

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={epoch_loss / max(len(train_loader), 1):.4f} | "
            f"val_loss={val_loss:.4f}"
        )

    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(val_x_tensor)).numpy().reshape(-1)
        test_probs = torch.sigmoid(model(test_x_tensor)).numpy().reshape(-1)

    best_threshold, _ = _find_best_threshold(val_y.reshape(-1).astype(int), val_probs)
    predictions = (test_probs >= best_threshold).astype(int)
    metrics = _classification_metrics(test_y.reshape(-1).astype(int), predictions)
    metrics["roc_auc"] = _roc_auc_score(test_y.reshape(-1).astype(int), test_probs)
    metrics["threshold"] = best_threshold
    importance_df = compute_permutation_importance(
        model=model,
        x=test_x,
        y=test_y.astype(int),
        feature_columns=feature_columns,
        baseline_metric=metrics["accuracy"],
    )
    top_feature_importance = importance_df.head(10).to_dict(orient="records")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": feature_columns,
            "train_mean": train_mean.tolist(),
            "train_std": train_std.tolist(),
            "metrics": metrics,
            "feature_importance": top_feature_importance,
            "threshold": best_threshold,
        },
        MODEL_FILE,
    )
    importance_df.to_csv(FEATURE_IMPORTANCE_FILE, index=False)
    MODEL_METRICS_FILE.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nTest Metrics")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    print(f"Model saved to: {MODEL_FILE}")
    print("\nTop Feature Importance")
    print(importance_df.head(10).to_string(index=False))
    logger.info("Model training completed | run_id=%s | metrics=%s", run_id, metrics)
    record_pipeline_run(
        run_id=run_id,
        stage="model_training",
        status="success",
        records_processed=len(df),
        issue_count=0,
        notes=f"Model trained and saved. Metrics: {metrics}. Top features: {top_feature_importance[:5]}",
    )


if __name__ == "__main__":
    main()
