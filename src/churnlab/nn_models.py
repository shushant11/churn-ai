"""Neural-network classifiers behind a scikit-learn interface. Your original Keras ANN is rebuilt PROPERLY (own validation split,
class weights, seeds) and a PyTorch twin is added. Both are optional: they only load if the library is installed."""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split


try:                                   # module-level so a fitted model can be pickled by joblib
    import torch
    import torch.nn as nn

    class _FTNet(nn.Module):
        """FT-Transformer style network for tabular data: every input column becomes a token (value * weight + bias),
        a learnable [CLS] token is prepended, a Transformer encoder mixes the tokens with self-attention,
        and the [CLS] output feeds a 1-unit head."""
        def __init__(self, n_feat, d_model, n_layers, n_heads, dropout):
            super().__init__()
            self.w = nn.Parameter(torch.randn(n_feat, d_model) * 0.02)
            self.b = nn.Parameter(torch.zeros(n_feat, d_model))
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2 * d_model, dropout=dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x):
            tok = x.unsqueeze(-1) * self.w + self.b                       # (batch, features, d_model)
            tok = torch.cat([self.cls.expand(x.shape[0], -1, -1), tok], dim=1)
            h = self.encoder(tok)[:, 0]                                    # [CLS] token
            return self.head(torch.relu(self.norm(h)))
except ImportError:                    # torch is optional
    _FTNet = None


def _fit_torch(net, X, y, lr, weight_decay, epochs, batch, patience, seed):
    """Shared training loop: internal stratified validation split (never the test set), class-weighted loss,
    early stopping on validation loss, best weights restored."""
    import torch, torch.nn as nn
    X = np.asarray(X, dtype="float32"); y = np.asarray(y, dtype="float32")
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.15, stratify=y, random_state=seed)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - yt.mean()) / yt.mean()))     # class imbalance handled in the loss
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    Xt_, yt_, Xv_, yv_ = map(torch.tensor, (Xt, yt, Xv, yv))
    best, bad, state = 1e9, 0, None
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt_))
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad(); loss_fn(net(Xt_[b]).squeeze(-1), yt_[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(Xv_).squeeze(-1), yv_).item()
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state); net.eval()
    return net


class _Base(ClassifierMixin, BaseEstimator):          # mixin FIRST, or scikit-learn >=1.6 treats the model as a regressor
    def predict_proba(self, X):
        p = self._predict(np.asarray(X, dtype="float32")).reshape(-1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class TorchANN(_Base):
    def __init__(self, hidden=(64, 32), lr=1e-3, epochs=100, batch=128, patience=10, dropout=0.2, weight_decay=1e-4, random_state=42):
        import torch  # noqa: F401  (raises ImportError if missing -> model is skipped)
        self.hidden, self.lr, self.epochs, self.batch, self.patience, self.dropout = hidden, lr, epochs, batch, patience, dropout
        self.weight_decay, self.random_state = weight_decay, random_state

    def fit(self, X, y):
        import torch, torch.nn as nn
        torch.manual_seed(self.random_state); np.random.seed(self.random_state)
        layers, d = [], np.asarray(X).shape[1]
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(self.dropout)]; d = h
        net = nn.Sequential(*layers, nn.Linear(d, 1))
        self.net_ = _fit_torch(net, X, y, self.lr, self.weight_decay, self.epochs, self.batch, self.patience, self.random_state)
        self.classes_ = np.array([0, 1])
        return self

    def _predict(self, X):
        import torch
        with torch.no_grad():
            return torch.sigmoid(self.net_(torch.tensor(X)).squeeze(-1)).numpy()


class TorchFT(_Base):
    """Tabular transformer (FT-Transformer style) - self-attention over the input columns."""
    def __init__(self, d_model=32, n_layers=2, n_heads=4, dropout=0.1, lr=1e-3, weight_decay=1e-4, epochs=60, batch=128, patience=8, random_state=42):
        import torch  # noqa: F401
        self.d_model, self.n_layers, self.n_heads, self.dropout = d_model, n_layers, n_heads, dropout
        self.lr, self.weight_decay, self.epochs, self.batch, self.patience, self.random_state = lr, weight_decay, epochs, batch, patience, random_state

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.random_state); np.random.seed(self.random_state)
        net = _FTNet(np.asarray(X).shape[1], self.d_model, self.n_layers, self.n_heads, self.dropout)
        self.net_ = _fit_torch(net, X, y, self.lr, self.weight_decay, self.epochs, self.batch, self.patience, self.random_state)
        self.classes_ = np.array([0, 1])
        return self

    def _predict(self, X):
        import torch
        with torch.no_grad():
            return torch.sigmoid(self.net_(torch.tensor(X)).squeeze(-1)).numpy()


class KerasANN(_Base):
    """Same 64-32-1 architecture as the original project, trained with a proper internal validation split."""
    def __init__(self, hidden=(64, 32), lr=1e-3, epochs=100, patience=10, dropout=0.0, random_state=42):
        import tensorflow  # noqa: F401
        self.hidden, self.lr, self.epochs, self.patience, self.dropout, self.random_state = hidden, lr, epochs, patience, dropout, random_state

    def fit(self, X, y):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.random_state)
        X = np.asarray(X, dtype="float32"); y = np.asarray(y, dtype="float32")
        layers = [tf.keras.layers.Input((X.shape[1],))]
        for h in self.hidden:
            layers.append(tf.keras.layers.Dense(h, activation="relu"))
            if self.dropout:
                layers.append(tf.keras.layers.Dropout(self.dropout))
        layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
        self.model_ = tf.keras.Sequential(layers)
        self.model_.compile(optimizer=tf.keras.optimizers.Adam(self.lr), loss="binary_crossentropy")
        w1 = float((1 - y.mean()) / y.mean())
        self.model_.fit(X, y, validation_split=0.15, epochs=self.epochs, batch_size=128, verbose=0, class_weight={0: 1.0, 1: w1},
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)])
        self.classes_ = np.array([0, 1])
        return self

    def _predict(self, X):
        return self.model_.predict(X, verbose=0).reshape(-1)
