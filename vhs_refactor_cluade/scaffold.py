"""
scaffold.py
Shared, fixed grid-hippocampal scaffold used by ALL FOUR memory models.

  Whg : fixed random projection, grid -> hippocampus (never trained)
  Wgh : learned ONCE via pseudoinverse over a training set of grid states,
        then frozen forever after; hippocampus -> grid

`cleanup` performs rounds of h -> g -> h to denoise a corrupted
hippocampal state back onto a valid scaffold attractor. This scaffold is
instantiated a single time and shared by ItemMemory / SpatialMemory /
SequenceMemory / MemoryPalace -- they differ only in what plastic
weights they layer on top of it.
"""
import numpy as np
from typing import List
from grid_utils import GridCode
from learning_rules import pseudoinverse_fit, hebbian_fit


class GridHPCScaffold:
    def __init__(self, grid_code: GridCode, Nh: int, seed: int = 0,
                 connection_prob: float = 1.0, threshold: float = 0.0,
                 nonlinearity: str = "relu_threshold", rule: str = "hebbian"):
        """
        connection_prob : target fraction of nonzero entries in Whg
            (paper's Wpg is a SPARSE random projection). 1.0 (default)
            leaves Whg dense. Initialization scale and pruning mechanics
            are matched to the original Fig3e notebook's
            `Wpg = randn(nruns, Np, Ng)` + randint-coordinate pruning
            (unnormalized entries, approximate -- not exact -- sparsity).
        threshold, nonlinearity : the guideline specifies
            h = ReLU(Whg . g - theta). nonlinearity="sign" (default)
            keeps this codebase's original h = sign(Whg . g), which is
            exact-pinv-friendly and is what the existing test suite
            assumes; nonlinearity="relu_threshold" switches to the
            literal formula (see config.PAPER_LIKE_SCAFFOLD).
        rule : Wgh (hpc -> grid) fitting rule, used once in `fit_wgh`.
            "hebbian" (default) matches every original notebook
            (fig2/3/4/6/7): Wgh = G @ H.T / P, a sum-of-outer-products
            that saturates/interferes as more states are stored -- this
            is what produces the paper's capacity phase-transition.
            "pinv" is the exact-recall alternative the original fig5
            notebook uses for its full-space Wgp (`gbook @ pinv(pbook)`)
            when only the sequence-transition capacity, not scaffold
            recall capacity, is under test.
        """
        assert nonlinearity in ("sign", "relu_threshold")
        assert rule in ("hebbian", "pinv")
        self.grid_code = grid_code
        self.Ng = grid_code.Ng
        self.Nh = Nh
        self.threshold = threshold
        self.nonlinearity = nonlinearity
        self.rule = rule
        rng = np.random.default_rng(seed)
        # Fixed random grid -> hippocampus projection (NEVER trained).
        # Matches the original Fig3e notebook's `Wpg = randn(nruns, Np, Ng)`:
        # unnormalized standard-normal entries (no 1/sqrt(Ng) scaling).
        self.Whg = rng.normal(0, 1.0, size=(Nh, self.Ng))
        # Pruning matches the original notebook's approach exactly:
        #   prune = int((1-c)*Np*Ng)
        #   mask[randint(low=0, high=Np, size=prune),
        #        randint(low=0, high=Ng, size=prune)] = 0
        # i.e. `prune` (row, col) coordinate pairs are drawn WITH
        # replacement and zeroed out -- collisions are not filtered, so
        # the number of entries actually zeroed can be slightly less than
        # `prune` and the realized sparsity is only approximately
        # `1 - connection_prob` (not an exact per-entry Bernoulli mask).
        prune = int((1.0 - connection_prob) * self.Nh * self.Ng)
        if prune > 0:
            mask = np.ones((self.Nh, self.Ng))
            row_idx = rng.integers(low=0, high=self.Nh, size=prune)
            col_idx = rng.integers(low=0, high=self.Ng, size=prune)
            mask[row_idx, col_idx] = 0
            self.Whg = self.Whg * mask
        self.Wgh = None  # learned once in `fit_wgh`, then frozen

    # ---------------- feedforward projections ----------------
    def grid_to_hpc(self, grid_state: np.ndarray) -> np.ndarray:
        logits = self.Whg @ grid_state
        if self.nonlinearity == "sign":
            return np.sign(logits)
        return np.maximum(logits - self.threshold, 0.0)

    def hpc_to_grid(self, hpc_state: np.ndarray) -> np.ndarray:
        assert self.Wgh is not None, "Call fit_wgh() once before using hpc_to_grid()."
        logits = self.Wgh @ hpc_state
        return self.grid_code.winner_take_all(logits)

    # ---------------- one-time scaffold learning ----------------
    def fit_wgh(self, grid_states: List[np.ndarray]) -> None:
        """Learn Wgh once over a sample of valid grid states (rule set
        in __init__), then freeze it for the rest of the scaffold's
        lifetime. This is a scaffold-level operation done a single
        time, shared by every downstream memory model."""
        G = np.stack(grid_states, axis=1)                             # (Ng, P)
        H = np.stack([self.grid_to_hpc(g) for g in grid_states], axis=1)  # (Nh, P)
        fit_fn = hebbian_fit if self.rule == "hebbian" else pseudoinverse_fit
        self.Wgh = fit_fn(H, G)                                        # G ~= Wgh @ H

    # ---------------- attractor cleanup ----------------
    def cleanup(self, hpc_state, n_iter: int = 1, recorder=None):
        h = hpc_state
        g = None
        for it in range(n_iter):   # max(1, n_iter) 제거
            g = self.hpc_to_grid(h)
            h = self.grid_to_hpc(g)
        return h, g