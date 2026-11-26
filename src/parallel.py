"""Reusable process-parallel helpers for CPU-bound per-sample work.

Design rules:
  * the worker function takes a small task id and returns a small result;
  * large read-only arrays live in module globals set before the pool starts,
    so Linux ``fork`` shares them copy-on-write (never pickle them per task);
  * each worker pins BLAS/torch to one thread, so 16 workers do not create
    16 x N threads and oversubscribe the cgroup quota;
  * choose the worker count from ``/sys/fs/cgroup/cpu.max`` (the rental
    quota), not from ``os.cpu_count()`` (the host).
"""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def cpu_quota() -> int:
    """Number of CPUs this cgroup may use (fallback: os.cpu_count())."""
    try:
        txt = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if txt and txt[0] != 'max':
            return max(1, int(float(txt[0]) / float(txt[1])))
    except Exception:
        pass
    return max(1, os.cpu_count() or 1)


def default_workers(reserve: int = 0) -> int:
    """Quota minus a small reserve for the parent/GPU feeder."""
    return max(1, cpu_quota() - int(reserve))


def worker_init() -> None:
    """Pin each worker to one BLAS/torch thread (avoid nested oversubscription)."""
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS'):
        os.environ[k] = '1'
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass


def parallel_map(func, tasks, workers: int | None = None, chunksize: int = 8):
    """Ordered map with a fork-based pool and one-thread workers.

    ``func`` and any large arrays must be resolvable in the child via fork
    (module-level globals or ``src.parallel._SHARED``).  Do not close over
    large arrays; set them as globals in the parent before calling.
    """
    if workers is None:
        workers = default_workers()
    tasks = list(tasks)
    if workers <= 1 or len(tasks) <= 1:
        worker_init()
        for t in tasks:
            yield func(t)
        return
    ctx = mp.get_context('fork')
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                             initializer=worker_init) as ex:
        yield from ex.map(func, tasks, chunksize=chunksize)
