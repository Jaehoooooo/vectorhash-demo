"""
datasets.py
Synthetic data generators shared across experiments.
"""
import numpy as np
from typing import List
import sys, os

def random_patterns(n_items: int, dim: int, seed: int = 0, bipolar: bool = True) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    if bipolar:
        data = rng.choice([-1.0, 1.0], size=(n_items, dim))
    else:
        data = rng.normal(size=(n_items, dim))
    return [row for row in data]


def add_noise(pattern: np.ndarray, noise_level: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    flip_mask = rng.random(pattern.shape) < noise_level
    noisy = pattern.copy()
    noisy[flip_mask] *= -1.0
    return noisy


def all_velocities(max_speed: int = 1) -> List[tuple]:
    """The full closed set of unit-step velocities random_walk_velocities
    samples from. Pass this as SequenceMemory's velocity_table so every
    velocity a training walk can produce has a matching output class."""
    return [(dx, dy) for dx in range(-max_speed, max_speed + 1)
            for dy in range(-max_speed, max_speed + 1) if not (dx == 0 and dy == 0)]


def random_walk_velocities(n_steps: int, max_speed: int = 1, seed: int = 0) -> List[tuple]:
    rng = np.random.default_rng(seed)
    choices = all_velocities(max_speed)
    idx = rng.integers(0, len(choices), size=n_steps)
    return [choices[i] for i in idx]


def prepare_sensory_data():
    """사용자가 제공한 MiniImageNet 전처리 로직 기반 데이터 로더"""
    # print("\n=== MiniImageNet 데이터 로드 및 전처리 시작 ===")
    
    # 파라미터 세팅
    Ns = 3600
    Npos = 60  
    n_states = Npos * Npos  # 3600
    
    block_x0 = 0
    block_y0 = 0
    block_w = 60
    block_h = 60
    
    # 1. 이미지 로드 (cwd에 상관없이 이 폴더(vhs_refactor_cluade) 안의 파일만 사용)
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'BW_miniimagenet_3600_60_60_full_rank.npy')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")
        
    img = np.load(file_path)
    # print("원본 이미지 형태:", img.shape)
    
    # 2. 평탄화 및 전치 (3600, 60, 60) -> (3600, 3600)
    img_flat = img.reshape((3600, 3600)).T
    
    # 3. Sensory book 초기화 및 이미지 블록 삽입
    sbook_flattened = np.random.randn(Ns, n_states)
    
    k = 0
    for x in range(block_x0, block_x0 + block_w):
        for y in range(block_y0, block_y0 + block_h):
            idx = x * Npos + y
            sbook_flattened[:, idx] = img_flat[:, k]
            k += 1
            
    # 4. 연상 기억을 위한 전처리 (스케일링 및 비선형 변환)
    bw_mean = np.mean(sbook_flattened)
    sbook_flattened = sbook_flattened - bw_mean
    
    sbookmin = np.amin(sbook_flattened)
    sbookmax = np.amax(sbook_flattened)
    
    sbook_scaled = np.interp(
        sbook_flattened,
        (sbookmin, sbookmax),
        (-0.95, +0.95)
    )
    
    sbookinv = np.arctanh(sbook_scaled)
    
    print(f"최종 sbook_flattened 형태: {sbookinv.shape}")
    print(f"값 범위: {np.min(sbookinv):.4f} ~ {np.max(sbookinv):.4f}")
    print("=== 데이터 전처리 완료 ===\n")
    
    return sbookinv