"""Training instrumentation: timers, memory probes, gradient norms.

These are called by the trainer at specific points (start/end of epoch,
after each optimizer step) to collect the raw numbers that our benchmark
reports.
"""
import time
import torch
import psutil
import os


# ── Timing ───────────────────────────────────────────────────────────

class Timer:
    """Simple wall-clock timer using time.perf_counter().

    perf_counter() is the most precise clock Python offers — it measures
    real elapsed time (not CPU time), which is what users actually
    experience.

    Usage:
        timer = Timer()
        timer.start()
        ... do work ...
        elapsed = timer.stop()  # returns seconds as float
    """

    def __init__(self):
        self._start = None
        self.elapsed = 0.0

    def start(self):
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            return 0.0
        self.elapsed = time.perf_counter() - self._start
        self._start = None
        return self.elapsed


# ── Memory ───────────────────────────────────────────────────────────

class MemoryProbe:
    """Measures peak memory usage on GPU and CPU.

    GPU: torch.cuda.max_memory_allocated() tracks the high-water mark
         of GPU memory used by tensors. We reset it at the start of
         each measurement window so we get per-epoch peaks.

    CPU: psutil reads the Resident Set Size (RSS) — how much physical
         RAM this Python process is using right now.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.has_cuda = device.type == "cuda"

    def reset_gpu_peak(self):
        """Reset the GPU peak counter to zero.

        Call this at the start of each epoch so peak_gpu_mb()
        reports the peak for *that epoch*, not the whole run.
        """
        if self.has_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

    def peak_gpu_mb(self) -> float:
        """Peak GPU memory allocated (in megabytes) since last reset."""
        if not self.has_cuda:
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

    def cpu_rss_mb(self) -> float:
        """Current CPU RAM usage of this process (in megabytes).

        RSS = Resident Set Size — the portion of this process's memory
        that is held in physical RAM right now.
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 2)


# ── Optimizer state size ─────────────────────────────────────────────

def optimizer_state_bytes(optimizer) -> int:
    """Total bytes used by the optimizer's internal state tensors.

    Adam stores two tensors (m and v) per parameter. SGD stores zero
    (or one if using momentum). This function walks all of them and
    sums their size in bytes.

    This is NOT the same as GPU memory — optimizer state might live on
    GPU or CPU. But it tells us the memory *overhead* of choosing one
    optimizer over another, independent of model size.
    """
    total = 0
    for param_state in optimizer.state.values():
        for val in param_state.values():
            if isinstance(val, torch.Tensor):
                total += val.element_size() * val.nelement()
    return total


# ── Gradient norms ───────────────────────────────────────────────────

def grad_norm(model) -> float:
    """Global L2 norm of all gradients in the model.

    After loss.backward(), every parameter with requires_grad=True has
    a .grad tensor. We flatten all of them into one vector and compute
    its L2 (Euclidean) length.

    Why this matters:
    - Exploding gradients: norm shoots up to 1000s → training diverges.
    - Vanishing gradients: norm drops to ~0 → model stops learning.
    - Healthy training: norm stays in a stable range.

    Our benchmark tracks this per step to detect stability issues.
    """
    grads = [
        p.grad.flatten()
        for p in model.parameters()
        if p.grad is not None
    ]
    if not grads:
        return 0.0
    return torch.cat(grads).norm(2).item()


def param_norm(model) -> float:
    """Global L2 norm of all parameter values.

    Useful paired with grad_norm — if param_norm grows without bound,
    the model weights are exploding (often caused by too-high lr).
    """
    params = [p.data.flatten() for p in model.parameters()]
    if not params:
        return 0.0
    return torch.cat(params).norm(2).item()


# ── Convergence speed ────────────────────────────────────────────────

def steps_to_threshold(curve: list, target: float, mode: str = "min") -> int:
    """Find the first step where a metric crosses a threshold.

    Parameters
    ----------
    curve : list of float
        The metric recorded at each step or epoch (e.g., test accuracy).
    target : float
        The threshold to reach (e.g., 0.97 for 97% accuracy).
    mode : "min" or "max"
        "min" — looking for curve[i] <= target (for loss).
        "max" — looking for curve[i] >= target (for accuracy / R²).

    Returns
    -------
    int
        The index (step number) where the threshold was first reached.
        Returns -1 if never reached — meaning the optimizer didn't
        converge to that level within the given training run.
    """
    for i, val in enumerate(curve):
        if mode == "max" and val >= target:
            return i
        if mode == "min" and val <= target:
            return i
    return -1