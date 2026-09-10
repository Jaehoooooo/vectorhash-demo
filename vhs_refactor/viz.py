"""
viz.py
Single shared visualization module for every experiment in this package.

Two independent tools live here:

  1. `savefig(fig, name)` -- save a plain matplotlib Figure as one named
     PDF+PNG under outputs/ (one file per sub-panel, matching the
     original paper notebooks' convention) and display it inline when
     running inside a notebook.

  2. `StateRecorder` -- record EVERY weight matrix and EVERY node-state
     vector of a model, once per real time-step, then render them all
     as one synchronized GIF. This is deliberately dumb: it never
     computes anything itself, it only stores whatever ndarrays the
     calling experiment hands it. The calling experiment must always
     get those ndarrays from actual model method calls (mem.recall(),
     scaffold.cleanup(), Wsh.recall(), ...) -- never by re-deriving the
     model's math by hand in the experiment file. That guarantee is the
     whole point: the animation can never silently drift out of sync
     with what the model actually does.

We deliberately do NOT call `matplotlib.use("Agg")` anywhere in this
module. Forcing a backend at import time would silently break inline
`plt.show()` display in Jupyter. `StateRecorder.render_gif` only ever
calls `anim.save(..., writer="pillow")`, which works under any backend,
headless or not, so nothing here needs Agg.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, Optional

STATIC_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")


# ===========================================================================
# 1. Static figures (one PDF+PNG per named sub-panel)
# ===========================================================================
def savefig(fig, name: str, show: bool = True) -> str:
    """Save `fig` as BOTH <name>.pdf and <name>.png under outputs/,
    display it inline (harmless no-op outside a notebook), then close
    it. Returns the PDF path."""
    os.makedirs(STATIC_OUT_DIR, exist_ok=True)
    pdf_path = os.path.join(STATIC_OUT_DIR, f"{name}.pdf")
    png_path = os.path.join(STATIC_OUT_DIR, f"{name}.png")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=150)
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  [saved] {pdf_path}")
    return pdf_path


def show_gif(path: str) -> None:
    """Display a saved GIF inline in a notebook. No-op outside one."""
    try:
        from IPython.display import Image, display
        display(Image(filename=path))
    except Exception:
        pass


# ===========================================================================
# 2. Real-time weight + state recorder
# ===========================================================================
def extract_matrix(obj) -> np.ndarray:
    """Pull a plain ndarray out of whatever a model's weight attribute
    actually is: a raw ndarray, a torch tensor, or a small wrapper class
    (HeteroAssociativeMemory, etc.) exposing it under a `.W` / `.weight`
    / `.weights` / `.matrix` attribute. Raises rather than guessing if
    none of these match, so a snapshot() call fails loudly instead of
    silently recording garbage."""
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "detach"):          # torch.Tensor
        return obj.detach().cpu().numpy()
    for attr in ("W", "weight", "weights", "matrix"):
        if hasattr(obj, attr):
            return extract_matrix(getattr(obj, attr))
    raise TypeError(
        f"extract_matrix: don't know how to get an ndarray out of {type(obj)}; "
        f"pass the raw array, a torch tensor, or add a .W/.weight/.weights/.matrix "
        f"attribute."
    )


class StateRecorder:
    """Call `.snapshot(...)` once per real time-step of an experiment;
    call `.render_gif(...)` once at the end to render every recorded
    weight matrix (heatmap) and node-state vector (line plot) as one
    synchronized animation.

    Usage:
        rec = StateRecorder()
        h = scaffold.grid_to_hpc(g)
        rec.snapshot(weights={"Whg": scaffold.Whg, "Wgh": scaffold.Wgh},
                     states={"g": g, "h": h}, label="t=0")
        for t, v in enumerate(velocities, start=1):
            g = grid_code.shift(g, v)
            h, g = scaffold.cleanup(scaffold.grid_to_hpc(g))
            rec.snapshot(weights={"Whg": scaffold.Whg, "Wgh": scaffold.Wgh},
                         states={"g": g, "h": h}, label=f"t={t}")
        rec.render_gif("outputs/live_demo.gif")

    Every value passed to `weights=`/`states=` must be something you got
    back from an actual model call -- never a hand-derived recomputation
    of what the model "should" produce.
    """

    def __init__(self):
        self.frames = []  # list of {"weights": {...}, "states": {...}, "label": str}

    def __len__(self) -> int:
        return len(self.frames)

    def snapshot(self, weights: Optional[Dict[str, object]] = None,
                 states: Optional[Dict[str, object]] = None, label: str = "") -> None:
        w = {k: extract_matrix(v) for k, v in (weights or {}).items()}
        s = {k: np.asarray(extract_matrix(v) if not isinstance(v, np.ndarray) else v,
                            dtype=float).ravel()
             for k, v in (states or {}).items()}
        self.frames.append({"weights": w, "states": s, "label": label})

    def render_gif(self, path: str, fps: float = 1.5, dpi: int = 100,
                    cmap: str = "RdBu_r", show: bool = True) -> str:
        assert self.frames, "StateRecorder.render_gif() called with no recorded frames"
        wnames = list(self.frames[0]["weights"].keys())
        snames = list(self.frames[0]["states"].keys())
        panels = wnames + snames
        n_panels = max(1, len(panels))
        ncols = min(4, n_panels)
        nrows = int(np.ceil(n_panels / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), dpi=dpi)
        axes = np.atleast_1d(axes).ravel()
        artists = {}

        for ax, name in zip(axes, panels):
            ax.set_title(name, fontsize=9)
            if name in wnames:
                w0 = self.frames[0]["weights"][name]
                vmax = max(float(np.abs(w0).max()), 1e-6)
                im = ax.imshow(w0, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
                artists[name] = ("weight", im, ax)
            else:
                s0 = self.frames[0]["states"][name]
                line, = ax.plot(np.arange(len(s0)), s0, lw=1.0, color="tab:blue")
                ax.set_xlim(-0.5, max(1, len(s0) - 0.5))
                artists[name] = ("state", line, ax)
        for ax in axes[len(panels):]:
            ax.axis("off")

        suptitle = fig.suptitle(self.frames[0]["label"], fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        def update(i):
            frame = self.frames[i]
            updated = []
            for name, (kind, artist, ax) in artists.items():
                if kind == "weight":
                    artist.set_data(frame["weights"][name])
                else:
                    y = frame["states"][name]
                    artist.set_ydata(y)
                    lo, hi = float(np.min(y)), float(np.max(y))
                    pad = 0.15 * (hi - lo + 1e-6)
                    ax.set_ylim(lo - pad, hi + pad)
                updated.append(artist)
            suptitle.set_text(frame["label"])
            return updated

        anim = animation.FuncAnimation(fig, update, frames=len(self.frames),
                                        interval=1000 / fps, blit=False)
        os.makedirs(STATIC_OUT_DIR, exist_ok=True)
        full_path = path if os.path.isabs(path) or os.sep in path \
            else os.path.join(STATIC_OUT_DIR, path)
        anim.save(full_path, writer="pillow", fps=fps)
        plt.close(fig)
        print(f"  [saved animation] {full_path}  ({len(self.frames)} frames, "
              f"{len(wnames)} weight panels, {len(snames)} state panels)")
        if show:
            show_gif(full_path)
        return full_path