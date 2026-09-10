"""
grid_utils.py
Multi-module toroidal grid code.

Each grid module m holds a 2D phase (x, y) living on a K_m x K_m torus.
The full grid state g is the concatenation of one-hot encodings of every
module's phase. A shift/velocity (vx, vy) is applied identically, in
physical units, to every module -- it wraps around each module's own
period. This is the mechanism that gives multi-module grid codes their
combinatorial capacity (Bienenstock/Fiete-style modular grid coding).
"""
import numpy as np
import itertools
from typing import List, Tuple

Velocity = Tuple[int, int]
ModuleIndices = List[Tuple[int, int]]


class GridCode:
    def __init__(self, module_periods: List[int], seed: int = 0):
        self.module_periods = list(module_periods)
        self.n_modules = len(module_periods)
        self.module_sizes = [k * k for k in module_periods]
        self.Ng = int(sum(self.module_sizes))
        self.C_s = int(np.prod([k * k for k in module_periods]))
        self.rng = np.random.default_rng(seed)

    # ---------------- encoding / decoding ----------------
    def encode_state(self, module_indices: ModuleIndices) -> np.ndarray:
        """module_indices: list of (x, y) per module -> flat one-hot grid vector g."""
        assert len(module_indices) == self.n_modules
        blocks = []
        for (x, y), k in zip(module_indices, self.module_periods):
            block = np.zeros((k, k), dtype=np.float64)
            block[x % k, y % k] = 1.0
            blocks.append(block.ravel())
        return np.concatenate(blocks)

    def decode_state(self, grid_state: np.ndarray) -> ModuleIndices:
        """Inverse of encode_state via per-module argmax."""
        idx = []
        offset = 0
        for k in self.module_periods:
            size = k * k
            block = grid_state[offset:offset + size].reshape(k, k)
            x, y = np.unravel_index(np.argmax(block), (k, k))
            idx.append((int(x), int(y)))
            offset += size
        return idx

    def state_blocks(self, grid_state: np.ndarray) -> List[np.ndarray]:
        """grid_state를 모듈별 k x k one-hot 블록 리스트로 분리 (시각화용)."""
        blocks = []
        offset = 0
        for k in self.module_periods:
            size = k * k
            blocks.append(grid_state[offset:offset + size].reshape(k, k))
            offset += size
        return blocks

    def random_state(self) -> ModuleIndices:
        return [(int(self.rng.integers(k)), int(self.rng.integers(k)))
                for k in self.module_periods]

    def all_states(self) -> List[ModuleIndices]:
        """Enumerate every combinatorial grid state (size C_s). Wgh must be
        fit over this full set (or the subset actually visited) for cleanup
        to be valid at every reachable grid state -- fitting on an arbitrary
        random sample only guarantees exact recovery for that sample, not
        for states a path-integration trajectory later visits."""
        per_module_options = [[(x, y) for x in range(k) for y in range(k)]
                               for k in self.module_periods]
        return [list(combo) for combo in itertools.product(*per_module_options)]

    # ---------------- dynamics ----------------
    def shift(self, grid_state: np.ndarray, velocity: Velocity) -> np.ndarray:
        """Grid transition T(g, v): identical physical shift on every module,
        wrapping mod K_m. Implemented with np.roll on each module's
        (K_m, K_m) one-hot block."""
        vx, vy = velocity
        blocks = []
        offset = 0
        for k in self.module_periods:
            size = k * k
            block = grid_state[offset:offset + size].reshape(k, k)
            shifted = np.roll(np.roll(block, vx % k, axis=0), vy % k, axis=1)
            blocks.append(shifted.ravel())
            offset += size
        return np.concatenate(blocks)

    def winner_take_all(self, logits: np.ndarray) -> np.ndarray:
        """Per-module argmax cleanup: continuous logits -> valid one-hot grid state."""
        out = np.zeros_like(logits)
        offset = 0
        for k in self.module_periods:
            size = k * k
            block = logits[offset:offset + size]
            out[offset + int(np.argmax(block))] = 1.0
            offset += size
        return out

    def path_integrate(self, start_indices: ModuleIndices,
                        velocities: List[Velocity]) -> List[np.ndarray]:
        """Return the list of grid states visited: [g_0, g_1, ..., g_T]."""
        g = self.encode_state(start_indices)
        path = [g]
        for v in velocities:
            g = self.shift(g, v)
            path.append(g)
        return path
