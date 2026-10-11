"""Three trainable spatial reliability backends.

All backends expose ``fit(wafers, observed_masks)`` and
``predict_proba(wafer, observed_mask)``.  The latter returns P(bad) for every
valid die, including unmeasured ones.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .data import WaferData
from .featurizer import SpatialFeaturizer


class SpatialBackend:
    name = "base"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)


class GPBoostBackend(SpatialBackend):
    """Two-stage GP reconstruction plus tabular boosted classification.

    LightGBM is used when installed.  HistGradientBoosting is a dependency-free
    fallback so this repository remains runnable from requirements.txt.
    """
    name = "gp_boost"

    def __init__(self, gp_length_scale: float = 3.0, random_state: int = 42):
        self.featurizer = SpatialFeaturizer(gp_length_scale)
        self.random_state = random_state
        self.classifier = None

    def fit(self, wafers: Sequence[WaferData], observed_masks: Sequence[np.ndarray]):
        X = np.vstack([self.featurizer.transform(w, m) for w, m in zip(wafers, observed_masks)])
        # bad is the positive class because escape detection is the business risk.
        y = np.concatenate([1 - w.labels for w in wafers])
        try:
            from lightgbm import LGBMClassifier
            self.classifier = LGBMClassifier(n_estimators=250, learning_rate=0.04,
                num_leaves=15, min_child_samples=30, class_weight="balanced",
                random_state=self.random_state, verbosity=-1)
            self.classifier.fit(X, y)
        except ImportError:
            self.classifier = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04,
                max_leaf_nodes=15, l2_regularization=1.0, random_state=self.random_state)
            # HistGradientBoosting has no class_weight constructor argument.
            n_bad = max(int(y.sum()), 1)
            n_good = max(int(len(y) - y.sum()), 1)
            weights = np.where(y == 1, len(y) / (2 * n_bad), len(y) / (2 * n_good))
            self.classifier.fit(X, y, sample_weight=weights)
        return self

    def predict_proba(self, wafer: WaferData, observed_mask: np.ndarray) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("backend is not fitted")
        return self.classifier.predict_proba(self.featurizer.transform(wafer, observed_mask))[:, 1]


def _torch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError("GNN and Transformer backends require torch; install requirements-ml.txt") from e


class _TorchBackend(SpatialBackend):
    """Shared wafer-wise optimisation loop; deliberately batch-size one.

    Wafers have differing valid-die counts, and batch-size one avoids padding
    being accidentally treated as a physical die.
    """
    epochs = 80
    learning_rate = 2e-3

    def _fit_torch(self, wafers, masks, build_model):
        torch = _torch()
        torch.manual_seed(42)
        first = self._tokens(wafers[0], masks[0])
        self.scaler = StandardScaler().fit(np.vstack([self._tokens(w, m) for w, m in zip(wafers, masks)]))
        self.input_dim = first.shape[1]
        self.model = build_model(self.input_dim)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        all_y = np.concatenate([1 - w.labels for w in wafers])
        pos_weight = float((len(all_y) - all_y.sum()) / max(all_y.sum(), 1))
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
        self.model.train()
        for _ in range(self.epochs):
            for wafer, mask in zip(wafers, masks):
                x = torch.tensor(self.scaler.transform(self._tokens(wafer, mask)), dtype=torch.float32)
                y = torch.tensor(1 - wafer.labels, dtype=torch.float32)
                opt.zero_grad()
                loss_fn(self.model(x, wafer.xy), y).backward()
                opt.step()
        return self

    def save(self, path: str | Path) -> None:
        """Save state dict rather than pickling a torch module.

        This keeps artifacts portable and avoids serialising executable Python
        code.  ``load_backend`` recreates the declared architecture.
        """
        torch = _torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"name": self.name, "params": self.__dict__ | {
            "model": None, "scaler": None
        }, "input_dim": self.input_dim, "scaler": self.scaler,
                    "state_dict": self.model.state_dict()}, path)

    def predict_proba(self, wafer: WaferData, observed_mask: np.ndarray) -> np.ndarray:
        torch = _torch()
        if not hasattr(self, "model"):
            raise RuntimeError("backend is not fitted")
        self.model.eval()
        x = torch.tensor(self.scaler.transform(self._tokens(wafer, observed_mask)), dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(self.model(x, wafer.xy)).cpu().numpy()


class GraphSAGEBackend(_TorchBackend):
    """Small distance-weighted GraphSAGE node classifier over a wafer graph."""
    name = "graphsage"

    def __init__(self, hidden_dim: int = 32, k_neighbors: int = 8):
        self.hidden_dim, self.k_neighbors = hidden_dim, k_neighbors

    def _tokens(self, wafer, mask):
        # GP estimates and uncertainty are useful node attributes; edges supply
        # the local spatial inductive bias.
        return SpatialFeaturizer().transform(wafer, mask)

    def _build_model(self, n_in):
        torch = _torch()
        outer = self
        class GraphSAGE(torch.nn.Module):
            def __init__(self, n_in):
                super().__init__()
                self.in_layer = torch.nn.Linear(n_in, outer.hidden_dim)
                self.out_layer = torch.nn.Linear(outer.hidden_dim * 2, 1)
            def forward(self, x, xy):
                h = torch.relu(self.in_layer(x))
                p = torch.tensor(xy, dtype=torch.float32, device=x.device)
                d = torch.cdist(p, p)
                d.fill_diagonal_(float("inf"))
                k = min(outer.k_neighbors, len(x) - 1)
                _, idx = torch.topk(d, k=k, largest=False)
                neigh = h[idx].mean(dim=1)
                return self.out_layer(torch.cat([h, neigh], dim=1)).squeeze(1)
        return GraphSAGE(n_in)

    def fit(self, wafers, observed_masks):
        return self._fit_torch(wafers, observed_masks, self._build_model)


class SpatialTransformerBackend(_TorchBackend):
    """Cross-attention from every die query to sparse measured die tokens."""
    name = "spatial_transformer"

    def __init__(self, hidden_dim: int = 48, heads: int = 4):
        self.hidden_dim, self.heads = hidden_dim, heads

    def _tokens(self, wafer, mask):
        return SpatialFeaturizer().token_features(wafer, mask)

    def _build_model(self, n_in):
        torch = _torch()
        outer = self
        class CrossAttention(torch.nn.Module):
            def __init__(self, n_in):
                super().__init__()
                self.token = torch.nn.Linear(n_in, outer.hidden_dim)
                self.query = torch.nn.Sequential(torch.nn.Linear(3, outer.hidden_dim), torch.nn.GELU(), torch.nn.Linear(outer.hidden_dim, outer.hidden_dim))
                self.attn = torch.nn.MultiheadAttention(outer.hidden_dim, outer.heads, batch_first=True)
                self.head = torch.nn.Sequential(torch.nn.LayerNorm(outer.hidden_dim), torch.nn.Linear(outer.hidden_dim, 1))
            def forward(self, x, xy):
                device = x.device
                pos = torch.tensor(xy, dtype=torch.float32, device=device)
                radial = torch.linalg.vector_norm(pos, dim=1, keepdim=True)
                q = self.query(torch.cat([pos, radial], dim=1)).unsqueeze(0)
                keys = self.token(x).unsqueeze(0)
                # Last raw-token column is the observed mask.  Only measured
                # sites are keys/values; every die remains a query.
                key_padding = (x[:, -1] < 0.5).unsqueeze(0)
                z, _ = self.attn(q, keys, keys, key_padding_mask=key_padding)
                return self.head(z.squeeze(0)).squeeze(1)
        return CrossAttention(n_in)

    def fit(self, wafers, observed_masks):
        return self._fit_torch(wafers, observed_masks, self._build_model)


def load_backend(path: str | Path) -> SpatialBackend:
    path = Path(path)
    # Torch artifacts are explicit dictionaries, identified without importing
    # torch for the sklearn-only GP+boost workflow.
    if path.suffix in {".pt", ".pth"}:
        torch = _torch()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        registry = {"graphsage": GraphSAGEBackend, "spatial_transformer": SpatialTransformerBackend}
        backend = registry[payload["name"]](**{
            k: v for k, v in payload["params"].items()
            if k in {"hidden_dim", "k_neighbors", "heads"}
        })
        backend.input_dim = payload["input_dim"]
        backend.scaler = payload["scaler"]
        backend.model = backend._build_model(backend.input_dim)
        backend.model.load_state_dict(payload["state_dict"])
        return backend
    with path.open("rb") as f:
        return pickle.load(f)
