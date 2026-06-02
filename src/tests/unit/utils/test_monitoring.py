from unittest.mock import patch

from src.utils.monitoring import get_cpu_usage, get_disk_usage, get_gpu_vram_usage, get_memory_usage, get_system_stats


@patch("psutil.cpu_percent", return_value=12.3)
def test_get_cpu_usage(mock_cpu):
    assert get_cpu_usage() == 12.3


@patch("psutil.virtual_memory")
def test_get_memory_usage(mock_vm):
    mock_vm.return_value.percent = 45.6
    assert get_memory_usage() == 45.6


@patch("subprocess.check_output", return_value="100, 1000\n")
def test_get_gpu_vram_usage_success(mock_sub):
    assert get_gpu_vram_usage() == 10.0


@patch("subprocess.check_output", side_effect=Exception("nvidia-smi not found"))
def test_get_gpu_vram_usage_fail(mock_sub):
    assert get_gpu_vram_usage() is None


@patch("subprocess.check_output", return_value="100, 0\n")
def test_get_gpu_vram_usage_zero_total(mock_sub):
    assert get_gpu_vram_usage() is None


@patch("shutil.disk_usage", return_value=(1000, 300, 700))
def test_get_disk_usage(mock_du):
    assert get_disk_usage() == 30.0


@patch("src.utils.monitoring.get_cpu_usage", return_value=1.0)
@patch("src.utils.monitoring.get_memory_usage", return_value=2.0)
@patch("src.utils.monitoring.get_gpu_vram_usage", return_value=3.0)
@patch("src.utils.monitoring.get_disk_usage", return_value=4.0)
def test_get_system_stats(m1, m2, m3, m4):
    stats = get_system_stats()
    assert stats["cpu"] == 1.0
    assert stats["memory"] == 2.0
    assert stats["gpu_vram"] == 3.0
    assert stats["disk"] == 4.0
