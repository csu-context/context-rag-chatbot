import logging
import shutil

import psutil

logger = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_cpu_usage():
    """Returns CPU usage in percentage."""
    return psutil.cpu_percent(interval=1)


def get_memory_usage():
    """Returns memory usage in percentage."""
    return psutil.virtual_memory().percent


def get_gpu_vram_usage():
    """Returns GPU VRAM usage in percentage, if available using PyTorch."""
    if not TORCH_AVAILABLE:
        return None

    try:
        if not torch.cuda.is_available():
            return None

        # GPU 0의 메모리 정보 가져오기 (free_memory, total_memory)
        free_memory, total_memory = torch.cuda.mem_get_info(0)

        if total_memory == 0:
            return None

        used_memory = total_memory - free_memory
        return (used_memory / total_memory) * 100
    except Exception as e:
        logger.error(f"Failed to get GPU VRAM usage via PyTorch: {e}")
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
