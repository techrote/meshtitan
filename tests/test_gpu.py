import pytest

from meshtitan.config import GPUConfig
from meshtitan.gpu import GPUError, GPUInfo, assert_gpu_contract, parse_gpu_csv, select_gpu


def titan() -> GPUInfo:
    return GPUInfo(2, "GPU-abc", "NVIDIA TITAN RTX", 24576, "7.5", "580.0")


def test_parse_and_select_gpu():
    fields = ["index", "uuid", "name", "memory.total", "compute_cap", "driver_version"]
    rows = parse_gpu_csv("2, GPU-abc, NVIDIA TITAN RTX, 24576, 7.5, 580.0\n", fields)
    assert rows[0]["name"] == "NVIDIA TITAN RTX"
    assert select_gpu([titan()], "GPU-abc") == titan()
    assert select_gpu([titan()], "2") == titan()


def test_contract_rejects_wrong_gpu():
    wrong = GPUInfo(0, "GPU-other", "NVIDIA Tesla V100", 16384, "7.0", "580")
    with pytest.raises(GPUError):
        assert_gpu_contract(wrong, GPUConfig())


def test_contract_accepts_titan():
    assert_gpu_contract(titan(), GPUConfig())
