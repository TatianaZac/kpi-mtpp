from __future__ import annotations

import json
import multiprocessing as mp
import os
import socket
import subprocess
import sys
import time
from multiprocessing import shared_memory
from pathlib import Path
from time import perf_counter

from utils import DEFAULT_SEED, mean, make_rng


_PIPE_STOP = None
_SHM_STOP = -1



def _pipe_worker(conn):
    while True:
        value = conn.recv()
        if value is _PIPE_STOP:
            conn.send({"stopped": True})
            break
        response = {
            "logged_by": "python_pipe_worker",
            "pid": os.getpid(),
            "value": int(value),
            "timestamp_ns": time.time_ns(),
        }
        conn.send(response)



def _shared_worker(cmd_conn, shm_name: str):
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = shm.buf
    while True:
        cmd = cmd_conn.recv()
        if cmd == "STOP":
            cmd_conn.send({"stopped": True})
            break
        value = int.from_bytes(buf[:8], "little", signed=True)
        if value == _SHM_STOP:
            cmd_conn.send({"stopped": True})
            break
        payload = {
            "logged_by": "python_shared_memory_worker",
            "pid": os.getpid(),
            "value": value,
            "timestamp_ns": time.time_ns(),
        }
        encoded = json.dumps(payload).encode("utf-8")
        size = len(encoded)
        buf[8:12] = size.to_bytes(4, "little", signed=False)
        buf[12:12 + size] = encoded
        cmd_conn.send("DONE")
    shm.close()



def benchmark_pipe(rounds: int = 1_000, seed: int = DEFAULT_SEED) -> dict:
    parent, child = mp.Pipe()
    proc = mp.Process(target=_pipe_worker, args=(child,))
    proc.start()
    rng = make_rng(seed)

    latencies = []
    sample = None
    started = perf_counter()
    for idx in range(rounds):
        value = rng.randint(1, 1_000_000)
        t0 = perf_counter()
        parent.send(value)
        response = parent.recv()
        dt = perf_counter() - t0
        latencies.append(dt)
        if idx == 0:
            sample = response
    total_elapsed = perf_counter() - started

    parent.send(_PIPE_STOP)
    parent.recv()
    proc.join(timeout=2)

    return {
        "title": "Python ↔ Python через multiprocessing.Pipe",
        "environment": "one_language",
        "method": "message_passing_pipe",
        "rounds": rounds,
        "avg_round_trip_ms": mean(latencies) * 1000,
        "min_round_trip_ms": min(latencies) * 1000,
        "max_round_trip_ms": max(latencies) * 1000,
        "total_time_s": total_elapsed,
        "sample": sample,
    }



def benchmark_shared_memory(rounds: int = 1_000, seed: int = DEFAULT_SEED) -> dict:
    shm = shared_memory.SharedMemory(create=True, size=4096)
    parent, child = mp.Pipe()
    proc = mp.Process(target=_shared_worker, args=(child, shm.name))
    proc.start()
    rng = make_rng(seed + 1)

    latencies = []
    sample = None
    started = perf_counter()
    try:
        for idx in range(rounds):
            value = rng.randint(1, 1_000_000)
            shm.buf[:8] = int(value).to_bytes(8, "little", signed=True)
            t0 = perf_counter()
            parent.send("READ")
            parent.recv()
            size = int.from_bytes(shm.buf[8:12], "little", signed=False)
            payload = bytes(shm.buf[12:12 + size]).decode("utf-8")
            response = json.loads(payload)
            dt = perf_counter() - t0
            latencies.append(dt)
            if idx == 0:
                sample = response
        total_elapsed = perf_counter() - started
        parent.send("STOP")
        parent.recv()
        proc.join(timeout=2)
    finally:
        shm.close()
        shm.unlink()

    return {
        "title": "Python ↔ Python через shared_memory",
        "environment": "one_language",
        "method": "shared_memory",
        "rounds": rounds,
        "avg_round_trip_ms": mean(latencies) * 1000,
        "min_round_trip_ms": min(latencies) * 1000,
        "max_round_trip_ms": max(latencies) * 1000,
        "total_time_s": total_elapsed,
        "sample": sample,
    }



def _wait_for_port(host: str, port: int, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.3):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"Node.js helper did not start on {host}:{port}")



def benchmark_socket_node(rounds: int = 500, port: int = 5050, seed: int = DEFAULT_SEED, node_script: str | Path = "node_helper.js") -> dict:
    script = Path(node_script).resolve()
    if not script.exists():
        raise FileNotFoundError(f"Node.js helper not found: {script}")

    proc = subprocess.Popen(
        ["node", str(script), str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        _wait_for_port("127.0.0.1", port, timeout=5.0)
        rng = make_rng(seed + 2)
        latencies = []
        sample = None
        started = perf_counter()
        with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
            file = sock.makefile("rwb")
            for idx in range(rounds):
                payload = {"value": rng.randint(1, 1_000_000)}
                t0 = perf_counter()
                file.write((json.dumps(payload) + "\n").encode("utf-8"))
                file.flush()
                line = file.readline()
                dt = perf_counter() - t0
                if not line:
                    raise RuntimeError("Node.js helper closed connection unexpectedly")
                response = json.loads(line.decode("utf-8"))
                latencies.append(dt)
                if idx == 0:
                    sample = response
            file.write((json.dumps({"stop": True}) + "\n").encode("utf-8"))
            file.flush()
            file.readline()
        total_elapsed = perf_counter() - started
    finally:
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.terminate()
            proc.wait(timeout=3)

    stdout = proc.stdout.read() if proc.stdout else ""
    stderr = proc.stderr.read() if proc.stderr else ""

    return {
        "title": "Python ↔ Node.js через TCP socket",
        "environment": "two_languages",
        "method": "message_passing_socket",
        "rounds": rounds,
        "avg_round_trip_ms": mean(latencies) * 1000,
        "min_round_trip_ms": min(latencies) * 1000,
        "max_round_trip_ms": max(latencies) * 1000,
        "total_time_s": total_elapsed,
        "sample": sample,
        "node_stdout_tail": stdout[-500:],
        "node_stderr_tail": stderr[-500:],
    }



def compare_ipc(pipe_rounds: int = 1_000, shm_rounds: int = 1_000, socket_rounds: int = 500, node_script: str | Path = "node_helper.js", return_results: bool = False):
    print("\nЗадача 2. Передача даних між процесами різного роду")
    print("Методи: Pipe, Shared Memory, TCP Socket (Python ↔ Node.js)")

    pipe_res = benchmark_pipe(rounds=pipe_rounds)
    print(f"Pipe         : avg {pipe_res['avg_round_trip_ms']:.4f} ms | total {pipe_res['total_time_s']:.4f} s")

    shm_res = benchmark_shared_memory(rounds=shm_rounds)
    print(f"SharedMemory : avg {shm_res['avg_round_trip_ms']:.4f} ms | total {shm_res['total_time_s']:.4f} s")

    socket_res = benchmark_socket_node(rounds=socket_rounds, node_script=node_script)
    print(f"Socket+Node  : avg {socket_res['avg_round_trip_ms']:.4f} ms | total {socket_res['total_time_s']:.4f} s")

    best = min(
        [pipe_res, shm_res, socket_res],
        key=lambda item: item["avg_round_trip_ms"],
    )

    results = {
        "ipc": {
            "title": "Передача числа між основним і допоміжним процесами",
            "pipe": pipe_res,
            "shared_memory": shm_res,
            "socket_node": socket_res,
            "best": {
                "method": best["method"],
                "title": best["title"],
                "avg_round_trip_ms": best["avg_round_trip_ms"],
            },
        }
    }
    if return_results:
        return results
    return None
