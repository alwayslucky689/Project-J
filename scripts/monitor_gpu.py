"""
Poll nvidia-smi once per second and log GPU VRAM + utilization.
Runs until Ctrl+C. Writes to data/gpu_log.csv.

Run this in a separate terminal WHILE you use the assistant.

Usage:
    python scripts/monitor_gpu.py
"""
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "data" / "gpu_log.csv"

QUERY = "timestamp,memory.used,memory.total,utilization.gpu,utilization.memory"
ARGS = [
    "nvidia-smi",
    f"--query-gpu={QUERY}",
    "--format=csv,noheader,nounits",
    "-l", "1",
]


def main():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {LOG_PATH}")
    print("Press Ctrl+C to stop.\n")

    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "vram_used_mb", "vram_total_mb",
                         "gpu_util_pct", "mem_util_pct"])
        f.flush()

        proc = subprocess.Popen(
            ARGS, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )

        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 5:
                    continue
                used, total, gpu_util, mem_util = parts[1], parts[2], parts[3], parts[4]
                ts = datetime.now().strftime("%H:%M:%S")
                writer.writerow([ts, used, total, gpu_util, mem_util])
                f.flush()
                print(f"[{ts}] VRAM {used}/{total} MB  GPU {gpu_util}%  Mem {mem_util}%")
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            proc.terminate()


if __name__ == "__main__":
    main()