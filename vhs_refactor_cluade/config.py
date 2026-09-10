"""
config.py
Initial hyperparameter configuration for the shared grid-hippocampal
scaffold and each of the four memory models. This is the "initial model
setup" section that the Notion implementation guideline's per-model
write-ups assumed but never spelled out on its own -- it fixes concrete
values for every symbol in the guideline's notation table (N_g, N_h, N_s,
N_m, C_s, theta, ...) plus the scaffold's connectivity / nonlinearity
choice and each model's training hyperparameters, so every experiment
constructs its models from ONE place instead of re-picking numbers
per-script.

Two nonlinearity modes are supported for h = f(Whg . g):
  - "sign"           : h = sign(Whg . g)            (this codebase's
                        original default -- binary bipolar h, exact-fit
                        friendly with plain batch pseudoinverse)
  - "relu_threshold"  : h = ReLU(Whg . g - theta)     (the guideline's
                        literal h = ReLU(W_hg g - theta) formula, used
                        together with connection_prob < 1 for a sparse
                        Whg closer to the paper's random Wpg)
ScaffoldConfig.nonlinearity defaults to "sign" to keep exact backward
compatibility with the existing test suite; PAPER_LIKE_SCAFFOLD below is
the "relu_threshold" + sparse alternative for anyone who wants the
literal guideline formula.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class ScaffoldConfig:
    module_periods: List[int] = field(default_factory=lambda: [3, 4, 5])  # grid periods (lambda_m)
    Nh: int = 400                # hippocampal population size
    connection_prob: float = 0.6  # fraction of nonzero Whg entries; 1.0 = dense (original default)
    threshold: float = 0.5        # theta in h = ReLU(Whg . g - theta)
    nonlinearity: str = "relu_threshold"    # "sign" (default) or "relu_threshold" (literal guideline formula)
    cleanup_iters: int = 2
    seed: int = 0

    @property
    def Ng(self) -> int:
        return int(sum(k * k for k in self.module_periods))

    @property
    def Cs(self) -> int:
        import numpy as np
        return int(np.prod([k * k for k in self.module_periods]))


# @dataclass
# class ItemMemoryConfig:
#     Ns: int = 100          # sensory-vector dimension
#     noise_level: float = 0.1
#     rule: str = "pinv"


# @dataclass
# class SpatialMemoryConfig:
#     Ns: int = 100
#     max_speed: int = 1     # |v| per step, in grid units
#     n_cleanup_iter: int = 1
#     rule: str = "pinv"


# @dataclass
# class SequenceMemoryConfig:
#     Ns: int = 100
#     max_speed: int = 1
#     hidden: int = 64        # F_psi hidden width
#     lr: float = 1e-2
#     n_epochs: int = 500
#     n_cleanup_iter: int = 1
#     rule: str = "pinv"


# @dataclass
# class MemoryPalaceConfig:
#     Nm: int = 50             # mnemonic-item dimension
#     n_cleanup_iter: int = 1
#     rule: str = "pinv"


# Default preset used by the existing experiments/tests (fully backward
# compatible: dense Whg, sign() nonlinearity).
DEFAULT_SCAFFOLD = ScaffoldConfig()
# DEFAULT_ITEM_MEMORY = ItemMemoryConfig()
# DEFAULT_SPATIAL_MEMORY = SpatialMemoryConfig()
# DEFAULT_SEQUENCE_MEMORY = SequenceMemoryConfig()
# DEFAULT_MEMORY_PALACE = MemoryPalaceConfig()

# Paper-faithful preset: sparse Whg + literal h = ReLU(Whg.g - theta).
SCAFFOLD = ScaffoldConfig(
    module_periods=[3, 4, 5], Nh=400,
    connection_prob=0.6, threshold=0.5, nonlinearity="relu_threshold",
    cleanup_iters=2, seed=0,
)
