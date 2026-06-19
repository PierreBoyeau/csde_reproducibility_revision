import inspect
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.config import RESULTS_DIR

# ---------------------------------------------------------------------------
# GPU selection helpers
# ---------------------------------------------------------------------------


@dataclass
class GPUStats:
    index: int
    free_mib: int
    total_mib: int
    utilization_pct: int
    n_processes: int

    @property
    def used_mib(self) -> int:
        return self.total_mib - self.free_mib

    @property
    def free_fraction(self) -> float:
        return self.free_mib / self.total_mib if self.total_mib else 0.0


def query_gpu_stats() -> list[GPUStats]:
    """Return one GPUStats per GPU by querying nvidia-smi."""
    gpu_out = (
        subprocess.run(
            "nvidia-smi --query-gpu=index,utilization.gpu,memory.free,memory.total "
            "--format=csv,noheader,nounits".split(),
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()
    )

    proc_out = (
        subprocess.run(
            "nvidia-smi --query-compute-apps=gpu_index --format=csv,noheader,nounits".split(),
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .splitlines()
    )
    process_counts: dict[int, int] = {}
    for line in proc_out:
        if line.strip():
            i = int(line.strip())
            process_counts[i] = process_counts.get(i, 0) + 1

    stats = []
    for row in gpu_out:
        idx, util, free, total = [x.strip() for x in row.split(",")]
        idx = int(idx)
        stats.append(
            GPUStats(
                index=idx,
                free_mib=int(free),
                total_mib=int(total),
                utilization_pct=int(util),
                n_processes=process_counts.get(idx, 0),
            )
        )
    return stats


def score_gpus(
    stats: list[GPUStats],
    w_free_memory: float = 0.5,
    w_utilization: float = 0.3,
    w_processes: float = 0.2,
) -> np.ndarray:
    """Score each GPU in [0, 1] — higher means more available.

    Three normalised signals are combined linearly:
    * free memory fraction  (higher → better)
    * 1 - utilisation / 100 (lower util → better)
    * 1 - process_count / max_processes (fewer processes → better)

    Parameters
    ----------
    w_free_memory, w_utilization, w_processes:
        Relative weights; they are re-normalised internally so they need not
        sum to 1.
    """
    free_frac = np.array([s.free_fraction for s in stats])
    idle_frac = 1.0 - np.array([s.utilization_pct for s in stats]) / 100.0
    n_proc = np.array([s.n_processes for s in stats], dtype=float)
    max_proc = n_proc.max() or 1.0
    proc_frac = 1.0 - n_proc / max_proc

    total_w = w_free_memory + w_utilization + w_processes
    scores = (
        w_free_memory * free_frac + w_utilization * idle_frac + w_processes * proc_frac
    ) / total_w
    return scores


def select_gpus(
    n_gpus: int = 1,
    w_free_memory: float = 0.5,
    w_utilization: float = 0.3,
    w_processes: float = 0.2,
) -> list[int]:
    """Pick the N least-busy GPUs and set CUDA_VISIBLE_DEVICES.

    Always selects exactly ``n_gpus`` GPUs (or all available if fewer exist),
    committing even when every GPU is busy — it simply picks the least-loaded
    ones according to a weighted score of free memory, utilisation, and active
    process count.

    Call this before importing JAX, PyTorch, or any other GPU-initialising
    library.

    Returns
    -------
    list[int]
        Selected GPU indices (also written to ``CUDA_VISIBLE_DEVICES``).
    """
    try:
        stats = query_gpu_stats()
        print(f"Found {len(stats)} GPU(s).")

        n_gpus = min(n_gpus, len(stats))
        scores = score_gpus(stats, w_free_memory, w_utilization, w_processes)
        ranked = np.argsort(scores)[::-1][:n_gpus]
        selected = [stats[i] for i in ranked]

        ids = [s.index for s in selected]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ids))
        for s in selected:
            print(
                f"  GPU {s.index}: score={scores[s.index]:.2f}, "
                f"free={s.free_mib}/{s.total_mib} MiB, "
                f"util={s.utilization_pct}%, procs={s.n_processes}"
            )
        return ids

    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"nvidia-smi unavailable; could not auto-select GPU. Error: {exc}")
        return []


def get_results_dir() -> str:
    caller_file = inspect.stack()[1].filename
    script_name = Path(caller_file).stem
    results_dir = os.path.join(RESULTS_DIR, script_name)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
