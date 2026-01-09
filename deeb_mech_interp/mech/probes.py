from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ProbeConfig
from .utils import normalize_frame, pick


class _VecDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _split_by_qid(meta: List[Dict[str, Any]], seed: int = 0, test_frac: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    qids = [str(pick(m, "qid", "id", "question_id")) for m in meta]
    uniq = sorted(set(qids))
    rnd = random.Random(seed)
    rnd.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_frac))
    test_q = set(uniq[:n_test])
    train_idx = np.array([i for i, q in enumerate(qids) if q not in test_q], dtype=np.int64)
    test_idx = np.array([i for i, q in enumerate(qids) if q in test_q], dtype=np.int64)
    return train_idx, test_idx


def _get_target(meta: List[Dict[str, Any]], target: str) -> Tuple[np.ndarray, List[str]]:
    t = target.strip().lower()
    labels = []
    for m in meta:
        if t == "frame":
            fr = normalize_frame(str(pick(m, "frame", "framing")))
            labels.append(fr)
        elif t in ("variant", "model_variant"):
            v = pick(m, "variant", "model_variant", "model", "model_id")
            if v is None:
                raise ValueError("Metadata missing `variant` (or `model_variant`) for probe target=variant.")
            labels.append(str(v))
        else:
            raise ValueError(f"Unsupported target: {target}. Use frame or variant.")
    classes = sorted(set(labels))
    cls_to_id = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_to_id[x] for x in labels], dtype=np.int64)
    return y, classes


def train_layerwise_probes(
    acts: np.ndarray,  # (N,L,H)
    meta: List[Dict[str, Any]],
    target: str = "frame",
    cfg: Optional[ProbeConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = ProbeConfig()

    N, L, H = acts.shape
    y, classes = _get_target(meta, target)
    train_idx, test_idx = _split_by_qid(meta, seed=cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report = {
        "target": target,
        "classes": classes,
        "n_samples": int(N),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "layers_total": int(L),
        "hidden_size": int(H),
        "layer_results": [],
    }

    for li in range(L):
        X = acts[:, li, :].astype(np.float32)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        ds_train = _VecDataset(X_train, y_train)
        ds_test = _VecDataset(X_test, y_test)

        dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
        dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

        model = nn.Linear(H, len(classes), bias=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(cfg.epochs):
            model.train()
            for xb, yb in dl_train:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in dl_test:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb).argmax(dim=-1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())
        acc = correct / max(1, total)

        report["layer_results"].append({"layer": li, "acc": float(acc)})

    # Best layer
    best = max(report["layer_results"], key=lambda r: r["acc"])
    report["best_layer"] = best
    return report
