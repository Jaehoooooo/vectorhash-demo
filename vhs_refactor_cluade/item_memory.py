"""
item_memory.py
Model 1: Item Memory  (Vector-HaSH paper, Fig. 1c)

No velocity, no transitions. Each item i is bound to an arbitrarily
assigned (random) grid codeword g_i -- items have no spatial relation to
one another. Learning stores plastic Whs / Wsh between sensory patterns
and their corresponding hippocampal states. Inference takes a noisy
sensory cue, projects it feedforward into hippocampus, cleans it up
through the fixed grid-hippocampal scaffold's cleanup loop, and reads
the cleaned sensory pattern back out.
"""
import numpy as np
from typing import List
from grid_utils import GridCode
from scaffold import GridHPCScaffold
from learning_rules import HeteroAssociativeMemory


class ItemMemory:
    def __init__(self, grid_code: GridCode, scaffold: GridHPCScaffold, Ns: int, rule: str = "pinv"):
        self.grid_code = grid_code
        self.scaffold = scaffold
        self.Ns = Ns
        self.Whs = HeteroAssociativeMemory(Ns, scaffold.Nh, rule=rule)  # sensory -> hpc
        self.Wsh = HeteroAssociativeMemory(scaffold.Nh, Ns, rule=rule)  # hpc -> sensory
        self.codewords: List[np.ndarray] = []
        self.hpc_states: List[np.ndarray] = []
        self._used_state_idxs: set = set()

    # ---------------- Learning phase ----------------
    def learn(self, sensory_items: List[np.ndarray]) -> None:
        # 원본 VectorHaSH의 gen_gbook(x -> x mod lambdas, src/assoc_utils_np.py)과
        # 같은 방식: item index를 그대로 codebook 순서에 매핑한다(무작위 추출 아님).
        # n <= C_s(전체 combinatorial grid state 수)면 자동으로 전부 다른 codeword를
        # 받고(중복 없음), n > C_s면 C_s 주기로 정확히 wrap-around 되어 item i와
        # item i+C_s가 같은 codeword를 공유한다 -- capacity를 넘었을 때의 결정론적
        # 충돌 패턴도 원본과 동일하게 재현된다.
        n = len(sensory_items)
        all_states = self.grid_code.all_states()
        idxs = np.arange(n) % len(all_states)
        self.codewords = [self.grid_code.encode_state(all_states[i]) for i in idxs]
        self.hpc_states = [self.scaffold.grid_to_hpc(g) for g in self.codewords]
        self._used_state_idxs = set(int(i) for i in idxs)
        self.Whs.fit(sensory_items, self.hpc_states)
        self.Wsh.fit(self.hpc_states, sensory_items)

    def add_item(self, sensory_item: np.ndarray) -> np.ndarray:
        """기존에 학습된 item들은 그대로 둔 채 새 sensory item 하나만 추가
        (scaffold의 Wgh는 건드리지 않고, Whs/Wsh만 HeteroAssociativeMemory.add()로
        1건 증분 학습). learn()과 같은 순서로 다음 codebook 자리(len(codewords) mod
        C_s)를 배정하고, 그 grid state(one-hot 인코딩)를 반환한다."""
        all_states = self.grid_code.all_states()
        idx = len(self.codewords) % len(all_states)
        self._used_state_idxs.add(idx)

        g = self.grid_code.encode_state(all_states[idx])
        h = self.scaffold.grid_to_hpc(g)
        self.codewords.append(g)
        self.hpc_states.append(h)
        self.Whs.add(sensory_item, h)
        self.Wsh.add(h, sensory_item)
        return g

    # ---------------- Inference phase ----------------
    def recall(self, noisy_sensory: np.ndarray, n_cleanup_iter: int = 2, recorder=None) -> np.ndarray:
        h0 = self.Whs.recall(noisy_sensory)                              # feedforward s -> h
        if recorder is not None:
            recorder.record(node_states={"h": h0}, weights={"Whs": self.Whs.W, "Wsh": self.Wsh.W},
                             label="feedforward s->h")
        h_clean, _ = self.scaffold.cleanup(h0, n_iter=n_cleanup_iter, recorder=recorder)  # scaffold cleanup loop
        return self.Wsh.recall(h_clean)                                  # h -> s readout

    def familiarity_error(self, sensory_cue: np.ndarray, n_cleanup_iter: int = 2) -> float:
        """Fig 3 novelty-detection signal: residual mismatch between the
        feedforward hippocampal projection h0 and the scaffold's cleaned
        attractor h_clean. A previously-stored item lands near a learned
        fixed point (small residual); a never-seen item does not."""
        h0 = self.Whs.recall(sensory_cue)
        h_clean, _ = self.scaffold.cleanup(h0, n_iter=n_cleanup_iter)
        return float(np.linalg.norm(h0 - h_clean) / (np.linalg.norm(h0) + 1e-12))