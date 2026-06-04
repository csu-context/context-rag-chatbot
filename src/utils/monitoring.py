import logging
import shutil
import subprocess

import psutil

logger = logging.getLogger(__name__)


def get_cpu_usage():
    """Returns CPU usage in percentage."""
    return psutil.cpu_percent(interval=None)


def get_memory_usage():
    """Returns memory usage in percentage."""
    return psutil.virtual_memory().percent


def get_gpu_vram_usage():
    """Returns GPU VRAM usage in percentage via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=3,
        )
        used, total = out.strip().splitlines()[0].split(",")
        total_mb = float(total.strip())
        if total_mb == 0:
            return None
        return (float(used.strip()) / total_mb) * 100
    except Exception:
        return None


def get_disk_usage():
    """Returns disk usage of the root directory in percentage."""
    total, used, _free = shutil.disk_usage("/")
    return (used / total) * 100


def get_system_stats():
    """Returns a dictionary of system stats."""
    return {
        "cpu": get_cpu_usage(),
        "memory": get_memory_usage(),
        "gpu_vram": get_gpu_vram_usage(),
        "disk": get_disk_usage(),
    }
