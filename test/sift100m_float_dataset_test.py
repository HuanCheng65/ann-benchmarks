from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import h5py
import numpy as np

from ann_benchmarks.datasets import DATASETS


def _load_converter_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "convert_sift100m_to_hdf5.py"
    spec = spec_from_file_location("convert_sift100m_to_hdf5", script_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sift100m_float_dataset_is_registered():
    assert "sift100m-128-euclidean-float" in DATASETS


def test_converter_can_write_float32_dataset(tmp_path):
    converter = _load_converter_module()

    train = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.uint8)
    test = np.array([[6, 7], [8, 9]], dtype=np.uint8)
    neighbors = np.array([[0, 1], [2, 1]], dtype=np.int32)
    distances = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    output = tmp_path / "sift100m-float.hdf5"

    converter.write_output_dataset(output, train, test, neighbors, distances, np.float32)

    with h5py.File(output, "r") as h5:
        assert h5.attrs["distance"] == "euclidean"
        assert h5.attrs["point_type"] == "float"
        assert h5["train"].dtype == np.float32
        assert h5["test"].dtype == np.float32
        np.testing.assert_array_equal(h5["neighbors"][:], neighbors)
        np.testing.assert_allclose(h5["distances"][:], distances)
