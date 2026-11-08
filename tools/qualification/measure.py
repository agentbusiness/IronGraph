#!/usr/bin/env python3
"""Measure an owned command with optional process-memory sampling; preserve failed output."""
import argparse
import ctypes
import json
import platform
import subprocess
import threading
import time
from pathlib import Path


def sampler():
    if platform.system() == "Darwin":
        # rusage_info_v2 from sys/resource.h: UUID followed by 18 uint64 fields.
        class Usage(ctypes.Structure):
            _fields_ = [("uuid", ctypes.c_ubyte * 16), ("values", ctypes.c_uint64 * 18)]

        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]

        def read(pid):
            usage = Usage()
            if lib.proc_pid_rusage(pid, 2, ctypes.byref(usage)) != 0:
                return None
            return {"rss_bytes": usage.values[6], "physical_bytes": usage.values[7]}

        return read
    if platform.system() == "Linux":
        def read(pid):
            try:
                rows = Path(f"/proc/{pid}/status").read_text().splitlines()
                rss = next(int(row.split()[1]) * 1024 for row in rows if row.startswith("VmRSS:"))
                return {"rss_bytes": rss, "physical_bytes": None}
            except (OSError, StopIteration):
                return None
        return read
    raise RuntimeError("process-memory sampling is supported on macOS/Linux")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-sampling", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("supply a command after --")
    read = sampler()
    samples = []
    events = []
    stopped = threading.Event()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    started_unix_ms = time.time_ns() / 1_000_000
    # File output avoids pipe backpressure changing the query workload.
    with args.output.with_suffix(".log").open("wb") as output:
        child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        def collect():
            while not stopped.is_set():
                value = read(child.pid)
                if value:
                    samples.append({"seconds": time.perf_counter() - start,
                                    "unix_ms": time.time_ns() / 1_000_000, **value})
                stopped.wait(0.01)
        worker = threading.Thread(target=collect)
        if not args.no_sampling:
            worker.start()
        try:
            code = child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
            stopped.set()
            if not args.no_sampling:
                worker.join()
    wall = time.perf_counter() - start
    for line in args.output.with_suffix(".log").read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
            if isinstance(event, dict):
                events.append(event)
        except json.JSONDecodeError:
            pass
    report = {"command": command, "exit_code": code, "wall_seconds": wall,
              "started_unix_ms": started_unix_ms,
              "sampling": not args.no_sampling, "sample_interval_ms": 10,
              "peak_rss_bytes": max((s["rss_bytes"] for s in samples), default=None),
              "peak_physical_bytes": max((s["physical_bytes"] for s in samples if s["physical_bytes"] is not None), default=None),
              "samples": samples, "events": events, "platform": platform.platform(),
              "boundary": "direct child process only; invoke compiled test binary, not cargo for test-process memory"}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in {"samples", "events", "command"}}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
