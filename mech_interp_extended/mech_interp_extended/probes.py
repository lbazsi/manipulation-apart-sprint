from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import ProbeConfig
from .runner import HFRunner


@dataclass
class LinearProbe:
    metric_col: str
    scaler: StandardScaler
    model: Ridge

    def predict_hidden(self, h: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(h)
        return self.model.predict(x)

    def predict_hidden_1(self, h1: np.ndarray) -> float:
        return float(self.predict_hidden(h1[None, :])[0])


def train_probe(
    df: pd.DataFrame,
    runner: HFRunner,
    cfg: ProbeConfig,
    metric_col: Optional[str] = None,
) -> Tuple[LinearProbe, dict]:
    """
    Train Ridge regression on final-layer last-token hidden to predict judge score column.
    """
    metric_col = metric_col or f"score_{cfg.metric_name}"
    if metric_col not in df.columns:
        raise ValueError(f"Missing metric column: {metric_col}")

    d = df[df[metric_col].notna()].copy()
    if len(d) == 0:
        raise ValueError(f"No non-NaN values for {metric_col}")

    if len(d) > cfg.max_train_rows:
        d = d.sample(cfg.max_train_rows, random_state=cfg.seed)

    prompts = d["prompt"].tolist()
    y = d[metric_col].astype(float).to_numpy()

    # Extract final hidden
    hs = runner.last_token_all_layers(prompts)  # (B, L+1, H)
    x = hs[:, -1, :].numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=cfg.train_frac, random_state=cfg.seed
    )

    scaler = StandardScaler(with_mean=True, with_std=True)
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    model = Ridge(alpha=cfg.ridge_alpha, random_state=cfg.seed)
    model.fit(x_train_s, y_train)

    pred = model.predict(x_test_s)
    mse = float(np.mean((pred - y_test) ** 2))
    r2 = float(model.score(x_test_s, y_test))

    probe = LinearProbe(metric_col=metric_col, scaler=scaler, model=model)
    report = {"metric_col": metric_col, "mse": mse, "r2": r2, "n_train": len(x_train), "n_test": len(x_test)}
    return probe, report