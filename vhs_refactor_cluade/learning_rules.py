"""
learning_rules.py
Generic plastic-weight learning rules and a reusable heteroassociative
memory module used by every model to bind one representation to another
(sensory<->hippocampus, sensory<->mnemonic item, etc). Every one of the
four memory models composes this rather than re-implementing fitting logic.
"""
import numpy as np
from typing import List


def pseudoinverse_fit(keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """keys: (Dk, P), values: (Dv, P). Returns W (Dv, Dk) s.t. values ~= W @ keys,
    exact on the training set whenever the keys are linearly independent
    (in particular whenever P <= Dk)."""
    return values @ np.linalg.pinv(keys)


def hebbian_fit(keys: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Sum-of-outer-products Hebbian rule, normalized by pattern count."""
    P = keys.shape[1]
    return (values @ keys.T) / P


class HeteroAssociativeMemory:
    """Key -> value associative memory with plastic weights W.

    Supports both batch fitting (`fit`) and one-shot incremental updates
    (`add`), which the Memory Palace model relies on to bind new mnemonic
    items without ever retraining the shared scaffold.
    """

    def __init__(self, key_dim: int, value_dim: int, rule: str = "pinv"):
        assert rule in ("pinv", "hebbian")
        self.rule = rule
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.W = np.zeros((value_dim, key_dim))
        self._keys: List[np.ndarray] = []
        self._values: List[np.ndarray] = []

    def fit(self, keys: List[np.ndarray], values: List[np.ndarray]) -> None:
        """Batch (re-)fit over a full list of (key, value) pairs."""
        self._keys, self._values = list(keys), list(values)
        K = np.stack(self._keys, axis=1)
        V = np.stack(self._values, axis=1)
        fit_fn = pseudoinverse_fit if self.rule == "pinv" else hebbian_fit
        self.W = fit_fn(K, V)

    def add(self, key: np.ndarray, value: np.ndarray) -> None:
        """One-shot incremental binding (re-fits over all pairs seen so far)."""
        self._keys.append(key)
        self._values.append(value)
        self.fit(self._keys, self._values)

    def recall(self, key: np.ndarray) -> np.ndarray:
        return self.W @ key

    # ---------------- Fig 6 lesion / MTT-style reinforcement ----------------
    def lesion(self, fraction: float, seed: int = 0) -> "HeteroAssociativeMemory":
        """Return a NEW HeteroAssociativeMemory whose key-side units have a
        random `fraction` permanently silenced (Fig 6n/6o hippocampal
        lesion): the corresponding COLUMNS of W are zeroed, so those key
        dimensions can no longer contribute to recall."""
        rng = np.random.default_rng(seed)
        n_silence = int(round(fraction * self.key_dim))
        silenced = rng.choice(self.key_dim, size=n_silence, replace=False)
        lesioned = HeteroAssociativeMemory(self.key_dim, self.value_dim, rule=self.rule)
        lesioned.W = self.W.copy()
        lesioned.W[:, silenced] = 0.0
        lesioned._keys, lesioned._values = list(self._keys), list(self._values)
        return lesioned

    def reinforce(self, fraction: float, seed: int = 0) -> "HeteroAssociativeMemory":
        """Fig 6o-style post-lesion relearning: re-fit W from scratch over
        the ORIGINAL (key, value) pairs but with the same key units
        silenced throughout -- i.e. consolidation around the surviving
        units, rather than simply masking the old (undamaged) weights."""
        rng = np.random.default_rng(seed)
        n_silence = int(round(fraction * self.key_dim))
        silenced = rng.choice(self.key_dim, size=n_silence, replace=False)
        lesioned_keys = []
        for k in self._keys:
            k_lesioned = k.copy()
            k_lesioned[silenced] = 0.0
            lesioned_keys.append(k_lesioned)
        reinforced = HeteroAssociativeMemory(self.key_dim, self.value_dim, rule=self.rule)
        reinforced.fit(lesioned_keys, self._values)
        return reinforced
