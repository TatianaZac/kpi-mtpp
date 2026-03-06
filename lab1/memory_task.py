# memory_task.py
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import shared_memory
from time import perf_counter

import numpy as np


def _time_it(fn, *args, **kwargs):
    # Допоміжна функція для вимірювання часу виконання
    t0 = perf_counter()
    res = fn(*args, **kwargs)
    return perf_counter() - t0, res


def transpose_numpy(a: np.ndarray) -> np.ndarray:
    # Транспонування через NumPy
    return a.T.copy()


def transpose_blocked_threaded(a: np.ndarray, workers: int, block: int = 512) -> np.ndarray:
    # Блочне транспонування з копіюванням блоків у кілька потоків
    n = a.shape[0]
    out = np.empty_like(a)

    def copy_block(i0: int, j0: int):
        # Копіюю блок (i0:i1, j0:j1) у відповідне місце у транспонованій матриці
        i1 = min(i0 + block, n)
        j1 = min(j0 + block, n)
        out[j0:j1, i0:i1] = a[i0:i1, j0:j1].T

    jobs = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i in range(0, n, block):
            for j in range(0, n, block):
                jobs.append(ex.submit(copy_block, i, j))
        for job in jobs:
            job.result()

    return out


# Процеси (shared memory), щоб не копіювати 10000x10000 через pickle

_SHM_IN = None
_SHM_OUT = None
_A = None
_OUT = None
_N = None
_DTYPE = None


def _proc_init(shm_in_name: str, shm_out_name: str, shape: tuple[int, int], dtype_str: str):
    # Ініціалізатор для процесів: підключається до shared memory і будую NumPy view
    global _SHM_IN, _SHM_OUT, _A, _OUT, _N, _DTYPE
    _SHM_IN = shared_memory.SharedMemory(name=shm_in_name)
    _SHM_OUT = shared_memory.SharedMemory(name=shm_out_name)
    _DTYPE = np.dtype(dtype_str)
    _N = int(shape[0])
    _A = np.ndarray(shape, dtype=_DTYPE, buffer=_SHM_IN.buf)
    _OUT = np.ndarray(shape, dtype=_DTYPE, buffer=_SHM_OUT.buf)


def _copy_block_proc(args):
    # args = (i0, j0, block)
    i0, j0, block = args
    n = _N
    i1 = min(i0 + block, n)
    j1 = min(j0 + block, n)
    _OUT[j0:j1, i0:i1] = _A[i0:i1, j0:j1].T
    return 1


def transpose_blocked_processed(a: np.ndarray, workers: int, block: int = 512) -> np.ndarray:
    # Блочне транспонування у кілька процесів без копіювання вхідної матриці між процесами.
    n = int(a.shape[0])
    shape = (n, n)
    dtype = a.dtype

    shm_in = shared_memory.SharedMemory(create=True, size=a.nbytes)
    shm_out = shared_memory.SharedMemory(create=True, size=a.nbytes)
    try:
        a_sh = np.ndarray(shape, dtype=dtype, buffer=shm_in.buf)
        out_sh = np.ndarray(shape, dtype=dtype, buffer=shm_out.buf)
        a_sh[:] = a

        jobs = [(i, j, block) for i in range(0, n, block) for j in range(0, n, block)]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_proc_init,
            initargs=(shm_in.name, shm_out.name, shape, dtype.str),
        ) as ex:
            for _ in ex.map(_copy_block_proc, jobs, chunksize=1):
                pass

        return out_sh.copy()
    finally:
        shm_in.close()
        shm_out.close()
        shm_in.unlink()
        shm_out.unlink()


def memory_demo(threads_list: list[int], size: int = 10000, return_results: bool = False):
    print("\n--- Memory-bound задача ---")
    print(f"Матриця: {size}x{size} (float32)")

    rng = np.random.default_rng(123)
    a = rng.random((size, size), dtype=np.float32)

    dt, out = _time_it(transpose_numpy, a)
    checksum = float(out[0, 0] + out[-1, -1])
    print(f"Послідовно (NumPy): час, c: {dt:.4f} | контрольна сума: {checksum:.6f}")

    results = {
        "memory": {
            "title": "Transpose matrix (blocked)",
            "seq": float(dt),
            "threads": {},
            "procs": {},
            "size": int(size),
            "checksum": float(checksum),
        }
    }

    for w in threads_list:
        if w <= 1:
            continue
        dt2, out2 = _time_it(transpose_blocked_threaded, a, w)
        checksum2 = float(out2[0, 0] + out2[-1, -1])
        print(f"Паралельно (потоків={w}): час, c: {dt2:.4f} | контрольна сума: {checksum2:.6f}")
        results["memory"]["threads"][int(w)] = float(dt2)

        dt3, out3 = _time_it(transpose_blocked_processed, a, w)
        checksum3 = float(out3[0, 0] + out3[-1, -1])
        print(f"Паралельно (процесів={w}): час, c: {dt3:.4f} | контрольна сума: {checksum3:.6f}")
        results["memory"]["procs"][int(w)] = float(dt3)

    if return_results:
        return results