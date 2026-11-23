"""A log folder per optimization run (optimization.start_log puts it in
OUTPUT_FOLDER/logs): the run's config at the start, then the gpu's state while it
runs. Every line goes to disk as it is written, so the log survives the machine
going down mid-run (the gpu driver taking Windows with it) and shows what led up to it.

  run.json     the run's settings (optimization.run_record); the app adds config.json,
               its page config, which the page can load again
  gpu.csv      every INTERVAL seconds, from a thread of its own (so it goes on while a
               step hangs): the pole and step reached, the PyTorch allocator's memory
               (allocated / reserved / peak since the run started), the whole gpu as
               nvidia-smi sees it (memory, load, temperature, power, clocks, what limits
               the clocks) and the system's RAM and commit charge
  events.log   loading, the models released and loaded, each pole, saved, errors -- each
               with the memory at that moment. In one app session, the "released" lines
               of successive runs show whether unloaded models gave their memory back.

nvidia-smi runs as its own process: no extra dependency, and it answers even when
CUDA in this process is stuck. When the gpu has fallen off the bus its error is logged
(smi_error) instead of the gpu's columns.
"""
import csv
import ctypes
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import torch

INTERVAL = 5.0  # seconds between gpu.csv rows

_SMI_FIELDS = ["memory.used", "memory.total", "utilization.gpu", "temperature.gpu", "power.draw",
               "clocks.gr", "clocks.mem", "pstate", "clocks_event_reasons.active"]
_SMI_COLUMNS = ["gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_util_pct", "temp_c", "power_w",
                "clock_gr_mhz", "clock_mem_mhz", "pstate", "clock_limits"]
COLUMNS = (["time", "elapsed_s", "pole", "step", "steps_per_s",
            "torch_allocated_mb", "torch_reserved_mb", "torch_peak_mb"]
           + _SMI_COLUMNS + ["ram_used_mb", "commit_used_mb", "smi_error"])

# nvidia-smi's clocks_event_reasons bits: why the clocks are below their maximum
_CLOCK_LIMITS = {0x1: "idle", 0x2: "app_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
                 0x10: "sync_boost", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake",
                 0x100: "display_clocks"}


def _clock_limits(mask: str) -> str:
    try:
        bits = int(mask, 16)
    except ValueError:
        return mask
    return "|".join(name for bit, name in _CLOCK_LIMITS.items() if bits & bit) or "none"


def nvidia_smi() -> tuple[dict[str, str], str]:
    """The gpu's state -> ({column: value}, "") or ({}, the error)"""
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(_SMI_FIELDS)}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {}, f"{type(exc).__name__}: {exc}"
    if out.returncode != 0:
        return {}, (out.stdout + out.stderr).strip().replace("\n", " ")
    values = dict(zip(_SMI_COLUMNS, (v.strip() for v in out.stdout.splitlines()[0].split(","))))
    values["clock_limits"] = _clock_limits(values.get("clock_limits", ""))
    return values, ""


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def system_memory() -> tuple[str, str]:
    """(RAM in use, commit charge) in MB; blank off Windows"""
    if sys.platform != "win32":
        return "", ""
    status = _MemoryStatus(dwLength=ctypes.sizeof(_MemoryStatus))
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return "", ""
    mb = 2 ** 20
    return (str((status.ullTotalPhys - status.ullAvailPhys) // mb),
            str((status.ullTotalPageFile - status.ullAvailPageFile) // mb))


class RunLog:
    """The log of one run in `folder`; `files` ({name: text}) are written first.
    event() for what happens, wrap() a run_pole step callback to follow the steps,
    close() at the end."""

    def __init__(self, folder: Path, files: dict[str, str], device):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (folder / name).write_text(text)
        self.cuda = torch.device(device).type == "cuda"
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.monotonic()
        self.pole, self.step = "", 0
        self._last = (self.t0, 0)
        self._lock = threading.Lock()
        self._csv_file = open(folder / "gpu.csv", "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._events = open(folder / "events.log", "w")
        self._write(self._csv_file, lambda: self._csv.writerow(COLUMNS))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        print(f"Log -> {folder}")

    @staticmethod
    def _write(file, write: Callable[[], None]) -> None:
        write()
        file.flush()
        os.fsync(file.fileno())  # on disk now: the next thing may be a crash

    def _torch_memory(self) -> list[str]:
        if not self.cuda:
            return ["", "", ""]
        mb = 2 ** 20
        return [str(torch.cuda.memory_allocated() // mb), str(torch.cuda.memory_reserved() // mb),
                str(torch.cuda.max_memory_allocated() // mb)]

    def _row(self) -> list[str]:
        now = time.monotonic()
        last_time, last_step = self._last
        rate = (self.step - last_step) / (now - last_time) if now > last_time and self.step >= last_step else 0.0
        self._last = (now, self.step)
        smi, error = nvidia_smi()
        return ([time.strftime("%H:%M:%S"), f"{now - self.t0:.0f}", self.pole, str(self.step), f"{rate:.2f}"]
                + self._torch_memory() + [smi.get(c, "") for c in _SMI_COLUMNS] + list(system_memory()) + [error])

    def _sample_loop(self):
        while not self._stop.is_set():
            row = self._row()
            with self._lock:
                if not self._csv_file.closed:
                    self._write(self._csv_file, lambda: self._csv.writerow(row))
            self._stop.wait(INTERVAL)

    def event(self, text: str) -> None:
        """A line in events.log, with the memory now"""
        allocated, reserved, _ = self._torch_memory()
        smi, error = nvidia_smi()
        ram, commit = system_memory()
        gpu = f"gpu {smi['gpu_mem_used_mb']}/{smi['gpu_mem_total_mb']} MB" if smi else f"gpu: {error}"
        line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} +{time.monotonic() - self.t0:.0f}s  {text}\n"
                f"    torch allocated {allocated} MB, reserved {reserved} MB | {gpu} | ram {ram} MB, commit {commit} MB\n")
        with self._lock:
            if not self._events.closed:
                self._write(self._events, lambda: self._events.write(line))

    def wrap(self, on_step: Callable) -> Callable:
        """on_step for run_pole that keeps the log's pole and step up to date, then calls on_step"""
        def logged(sign, step, canvas, views, stats):
            pole = "+" if sign > 0 else "-"
            if pole != self.pole:
                self.event(f"pole {pole} started")
            self.pole, self.step = pole, step
            return on_step(sign, step, canvas, views, stats)
        return logged

    def close(self, status: str) -> None:
        """The last event (`status`: done / stopped / the error) and a last gpu.csv row"""
        self._stop.set()
        self._thread.join(timeout=30)
        self.event(status)
        row = self._row()
        with self._lock:
            self._write(self._csv_file, lambda: self._csv.writerow(row))
            self._csv_file.close()
            self._events.close()
