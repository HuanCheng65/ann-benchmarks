import numpy as np
import h5py

from ann_benchmarks import runner
from ann_benchmarks.algorithms.base.module import BaseANN


class CountingTrainArray:
    def __init__(self, data):
        self._data = np.asarray(data)
        self.shape = self._data.shape
        self.dtype = self._data.dtype
        self.requests = []

    def __getitem__(self, item):
        self.requests.append(item)
        return np.asarray(self._data[item])


def test_compute_candidates_with_distances_batches_train_reads():
    train = CountingTrainArray(
        [
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
        ]
    )
    queries = np.asarray([[0, 0], [3, 3]], dtype=np.uint8)
    results = [[4, 1, 4], [3, 2]]

    candidates = runner.compute_candidates_with_distances(train, queries, results, "euclidean")

    assert len(train.requests) == 1
    np.testing.assert_array_equal(train.requests[0], np.asarray([1, 2, 3, 4], dtype=np.int64))
    assert [idx for idx, _ in candidates[0]] == [4, 1, 4]
    assert [idx for idx, _ in candidates[1]] == [3, 2]
    np.testing.assert_allclose(candidates[0][0][1], np.sqrt(32.0))
    np.testing.assert_allclose(candidates[0][1][1], np.sqrt(2.0))
    np.testing.assert_allclose(candidates[1][0][1], 0.0)
    np.testing.assert_allclose(candidates[1][1][1], np.sqrt(2.0))


def test_load_and_transform_dataset_lazy_prefers_external_u8bin_override(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset.hdf5"
    base_path = tmp_path / "base.bin"
    train = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.uint8)
    test = np.asarray([[7, 8]], dtype=np.uint8)

    with h5py.File(dataset_path, "w") as h5:
        h5.attrs["distance"] = "euclidean"
        h5.attrs["point_type"] = "uint8"
        h5.create_dataset("train", data=train)
        h5.create_dataset("test", data=test)

    with open(base_path, "wb") as f:
        f.write(np.asarray([train.shape[0], train.shape[1]], dtype=np.int32).tobytes())
        f.write(train.tobytes(order="C"))

    monkeypatch.setenv("ANNB_EXTERNAL_BASE_U8BIN", str(base_path))

    def fake_get_dataset(_):
        handle = h5py.File(dataset_path, "r")
        return handle, train.shape[1]

    monkeypatch.setattr(runner, "get_dataset", fake_get_dataset)

    X_train, X_test, distance = runner.load_and_transform_dataset_lazy("dummy")

    assert isinstance(X_train, runner.LazyU8BinArray)
    np.testing.assert_array_equal(np.asarray(X_train[2]), train[2])
    np.testing.assert_array_equal(X_test, test)
    assert distance == "euclidean"


class NativeDistanceAlgo(BaseANN):
    def __init__(self):
        self.name = "NativeDistanceAlgo"
        self._results = None
        self._distances = None
        self._latencies = None

    def batch_query(self, X, n):
        self._results = [[4, 1], [3, 2]]
        self._distances = [[5.0, 1.0], [0.0, 2.0]]
        self._latencies = [0.25, 0.25]

    def get_batch_results(self):
        return self._results

    def get_batch_distances(self):
        return self._distances

    def get_batch_latencies(self):
        return self._latencies


def test_run_individual_query_uses_algorithm_provided_batch_distances(monkeypatch):
    algo = NativeDistanceAlgo()
    train = np.asarray([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=np.uint8)
    test = np.asarray([[0, 0], [3, 3]], dtype=np.uint8)

    monkeypatch.setattr(
        runner,
        "compute_candidates_with_distances",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("distance fallback should stay unused")),
    )

    descriptor, results = runner.run_individual_query(
        algo,
        train,
        test,
        "euclidean",
        count=2,
        run_count=1,
        batch=True,
    )

    assert descriptor["best_search_time"] == 0.25
    assert results == [
        (0.25, [(4, 5.0), (1, 1.0)]),
        (0.25, [(3, 0.0), (2, 2.0)]),
    ]
