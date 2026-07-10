from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def compute_permutation_importance(
    model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    feature_columns: list[str],
    baseline_metric: float,
) -> pd.DataFrame:
    importances: list[dict[str, float]] = []
    x_tensor_template = torch.tensor(x, dtype=torch.float32)

    for feature_index, feature_name in enumerate(feature_columns):
        permuted = x.copy()
        shuffled = permuted[:, feature_index].copy()
        np.random.default_rng(42 + feature_index).shuffle(shuffled)
        permuted[:, feature_index] = shuffled

        with torch.no_grad():
            probs = torch.sigmoid(model(torch.tensor(permuted, dtype=torch.float32))).numpy().reshape(-1)
        predictions = (probs >= 0.5).astype(int)
        accuracy = float((predictions == y.reshape(-1)).mean())
        importances.append(
            {
                "feature": feature_name,
                "importance": baseline_metric - accuracy,
            }
        )

    return pd.DataFrame(importances).sort_values("importance", ascending=False).reset_index(drop=True)
