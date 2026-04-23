import argparse
import json
import logging
import os
import threading
import time
from typing import Dict, Optional, Tuple, List, Union

import colors
import docker
import numpy
import psutil
import h5py

from ann_benchmarks.algorithms.base.module import BaseANN

from .definitions import Definition, instantiate_algorithm
from .datasets import DATASETS, get_dataset
from .distance import dataset_transform, metrics
from .results import store_results


class LazyHDF5Array:
    def __init__(self, dataset: h5py.Dataset):
        self._dataset = dataset
        self.shape = dataset.shape
        self.dtype = dataset.dtype

    def __len__(self):
        return int(self.shape[0])

    def __getitem__(self, item):
        value = self._dataset[item]
        if isinstance(item, slice):
            return numpy.asarray(value)
        return numpy.asarray(value)

    def __array__(self):
        return numpy.asarray(self._dataset)


class LazyU8BinArray:
    def __init__(self, path: str, count: int, dims: int):
        self._path = path
        self.shape = (int(count), int(dims))
        self.dtype = numpy.dtype(numpy.uint8)
        self._memmap = numpy.memmap(
            path,
            dtype=numpy.uint8,
            mode="r",
            offset=8,
            shape=self.shape,
            order="C",
        )

    def __len__(self):
        return int(self.shape[0])

    def __getitem__(self, item):
        return numpy.asarray(self._memmap[item])

    def __array__(self):
        return numpy.asarray(self._memmap)


def compute_candidates_with_distances(
    X_train: numpy.array,
    X_test: numpy.array,
    results: List[List[int]],
    distance: str,
) -> List[List[Tuple[int, float]]]:
    unique_ids = sorted({int(idx) for row in results for idx in row})
    if not unique_ids:
        return [[] for _ in results]

    train_batch = numpy.asarray(X_train[numpy.asarray(unique_ids, dtype=numpy.int64)])
    id_to_pos = {idx: pos for pos, idx in enumerate(unique_ids)}
    row_lengths = [len(row) for row in results]
    flat_ids = numpy.asarray([int(idx) for row in results for idx in row], dtype=numpy.int64)
    flat_query_ids = numpy.repeat(numpy.arange(len(results), dtype=numpy.int64), row_lengths)
    flat_train = train_batch[numpy.asarray([id_to_pos[int(idx)] for idx in flat_ids], dtype=numpy.int64)]
    flat_queries = numpy.asarray(X_test)[flat_query_ids]

    if distance == "euclidean":
        delta = flat_queries.astype(numpy.float32, copy=False) - flat_train.astype(numpy.float32, copy=False)
        flat_distances = numpy.sqrt(numpy.sum(delta * delta, axis=1, dtype=numpy.float32))
    elif distance == "angular":
        q = flat_queries.astype(numpy.float32, copy=False)
        t = flat_train.astype(numpy.float32, copy=False)
        numerator = numpy.sum(q * t, axis=1, dtype=numpy.float32)
        denominator = numpy.linalg.norm(q, axis=1) * numpy.linalg.norm(t, axis=1)
        flat_distances = 1.0 - numpy.divide(
            numerator,
            denominator,
            out=numpy.zeros_like(numerator),
            where=denominator != 0,
        )
    elif distance == "hamming":
        flat_distances = numpy.mean(
            numpy.logical_xor(
                flat_queries.astype(numpy.bool_, copy=False),
                flat_train.astype(numpy.bool_, copy=False),
            ),
            axis=1,
        )
    else:
        flat_distances = numpy.asarray(
            [metrics[distance].distance(a, b) for a, b in zip(flat_queries, flat_train)],
            dtype=numpy.float32,
        )

    candidates = []
    offset = 0
    for row in results:
        row_size = len(row)
        row_distances = flat_distances[offset : offset + row_size]
        candidates.append(
            [(int(idx), float(dist)) for idx, dist in zip(row, row_distances)]
        )
        offset += row_size
    return candidates


def run_individual_query(algo: BaseANN, X_train: numpy.array, X_test: numpy.array, distance: str, count: int, 
                         run_count: int, batch: bool) -> Tuple[dict, list]:
    """Run a search query using the provided algorithm and report the results.

    Args:
        algo (BaseANN): An instantiated ANN algorithm.
        X_train (numpy.array): The training data.
        X_test (numpy.array): The testing data.
        distance (str): The type of distance metric to use.
        count (int): The number of nearest neighbors to return.
        run_count (int): The number of times to run the query.
        batch (bool): Flag to indicate whether to run in batch mode or not.

    Returns:
        tuple: A tuple with the attributes of the algorithm run and the results.
    """
    prepared_queries = (batch and hasattr(algo, "prepare_batch_query")) or (
        (not batch) and hasattr(algo, "prepare_query")
    )

    best_search_time = float("inf")
    for i in range(run_count):
        print("Run %d/%d..." % (i + 1, run_count))
        # a bit dumb but can't be a scalar since of Python's scoping rules
        n_items_processed = [0]

        def single_query(v: numpy.array) -> Tuple[float, List[Tuple[int, float]]]:
            """Executes a single query on an instantiated, ANN algorithm.

            Args:
                v (numpy.array): Vector to query.

            Returns:
                List[Tuple[float, List[Tuple[int, float]]]]: Tuple containing
                    1. Total time taken for each query 
                    2. Result pairs consisting of (point index, distance to candidate data )
            """
            if prepared_queries:
                algo.prepare_query(v, count)
                start = time.time()
                algo.run_prepared_query()
                total = time.time() - start
                candidates = algo.get_prepared_query_results()
            else:
                start = time.time()
                candidates = algo.query(v, count)
                total = time.time() - start

            # make sure all returned indices are unique
            assert len(candidates) == len(set(candidates)), "Implementation returned duplicated candidates"

            candidates = compute_candidates_with_distances(X_train, numpy.asarray([v]), [candidates], distance)[0]
            n_items_processed[0] += 1
            if n_items_processed[0] % 1000 == 0:
                print("Processed %d/%d queries..." % (n_items_processed[0], len(X_test)))
            if len(candidates) > count:
                print(
                    "warning: algorithm %s returned %d results, but count"
                    " is only %d)" % (algo, len(candidates), count)
                )
            return (total, candidates)

        def batch_query(X: numpy.array) -> List[Tuple[float, List[Tuple[int, float]]]]:
            """Executes a batch of queries on an instantiated, ANN algorithm.

            Args:
                X (numpy.array): Array containing multiple vectors to query.

            Returns:
                List[Tuple[float, List[Tuple[int, float]]]]: List of tuples, each containing
                    1. Total time taken for each query 
                    2. Result pairs consisting of (point index, distance to candidate data )
            """
            # TODO: consider using a dataclass to represent return value.
            if prepared_queries:
                algo.prepare_batch_query(X, count)
                start = time.time()
                algo.run_batch_query()
                total = time.time() - start
            else:
                start = time.time()
                algo.batch_query(X, count)
                total = time.time() - start
            results = algo.get_batch_results()
            native_distances = algo.get_batch_distances() if hasattr(algo, "get_batch_distances") else None
            if hasattr(algo, "get_batch_latencies"):
                batch_latencies = algo.get_batch_latencies()
            else:
                batch_latencies = [total / float(len(X))] * len(X)

            # make sure all returned indices are unique
            for res in results:
                assert len(res) == len(set(res)), "Implementation returned duplicated candidates"

            if native_distances is not None:
                candidates = [
                    [(int(idx), float(dist)) for idx, dist in zip(single_results, single_distances)]
                    for single_results, single_distances in zip(results, native_distances)
                ]
            else:
                candidates = compute_candidates_with_distances(X_train, X, results, distance)
            return [(latency, v) for latency, v in zip(batch_latencies, candidates)]

        if batch:
            results = batch_query(X_test)
        else:
            results = [single_query(x) for x in X_test]

        total_time = sum(time for time, _ in results)
        total_candidates = sum(len(candidates) for _, candidates in results)
        search_time = total_time / len(X_test)
        avg_candidates = total_candidates / len(X_test)
        best_search_time = min(best_search_time, search_time)

    verbose = hasattr(algo, "query_verbose")
    attrs = {
        "batch_mode": batch,
        "best_search_time": best_search_time,
        "candidates": avg_candidates,
        "expect_extra": verbose,
        "name": str(algo),
        "run_count": run_count,
        "distance": distance,
        "count": int(count),
    }
    additional = algo.get_additional()
    for k in additional:
        attrs[k] = additional[k]
    return (attrs, results)


def load_and_transform_dataset(dataset_name: str) -> Tuple[
        Union[numpy.ndarray, List[numpy.ndarray]],
        Union[numpy.ndarray, List[numpy.ndarray]],
        str]:
    """Loads and transforms the dataset.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        Tuple: Transformed datasets.
    """
    D, dimension = get_dataset(dataset_name)
    X_train = numpy.array(D["train"])
    X_test = numpy.array(D["test"])
    distance = D.attrs["distance"]

    print(f"Got a train set of size ({X_train.shape[0]} * {dimension})")
    print(f"Got {len(X_test)} queries")

    train, test = dataset_transform(D)
    return train, test, distance


def load_and_transform_dataset_lazy(dataset_name: str) -> Tuple[
        Union[numpy.ndarray, List[numpy.ndarray], LazyHDF5Array, LazyU8BinArray],
        Union[numpy.ndarray, List[numpy.ndarray]],
        str]:
    D, dimension = get_dataset(dataset_name)
    distance = D.attrs["distance"]
    point_type = D.attrs.get("point_type", "float")
    external_base_override = os.environ.get("ANNB_EXTERNAL_BASE_U8BIN")

    test_source = D["test"]
    if external_base_override:
        train_shape = (
            int(D["train"].shape[0]),
            int(D["train"].shape[1]),
        )
        X_train = LazyU8BinArray(external_base_override, *train_shape)
    elif "external_base_u8bin" in D.attrs:
        train_shape = (
            int(D.attrs["external_train_count"]),
            int(D.attrs["external_train_dimension"]),
        )
        X_train = LazyU8BinArray(str(D.attrs["external_base_u8bin"]), *train_shape)
    else:
        train_source = D["train"]
        use_lazy_train = (
            dataset_name == "sift1b-128-euclidean"
            or int(train_source.shape[0]) >= 100_000_000
        )
        X_train = LazyHDF5Array(train_source) if use_lazy_train else numpy.array(train_source)
        train_shape = train_source.shape
    X_test = numpy.array(test_source)

    print(f"Got a train set of size ({train_shape[0]} * {dimension})")
    print(f"Got {len(X_test)} queries")

    if D.attrs.get("type", "dense") == "sparse":
        train, test = dataset_transform(D)
        return train, test, distance

    if point_type == "float":
        return X_train, X_test.astype(numpy.float32, copy=False), distance
    return X_train, X_test.astype(numpy.uint8, copy=False), distance


def build_index(algo: BaseANN, X_train: numpy.ndarray) -> Tuple:
    """Builds the ANN index for a given ANN algorithm on the training data.

    Args:
        algo (Any): The algorithm instance.
        X_train (Any): The training data.

    Returns:
        Tuple: The build time and index size.
    """
    t0 = time.time()
    memory_usage_before = algo.get_memory_usage()
    algo.fit(X_train)
    build_time = time.time() - t0
    index_size = algo.get_memory_usage() - memory_usage_before

    print("Built index in", build_time)
    print("Index size: ", index_size)

    return build_time, index_size


def run(definition: Definition, dataset_name: str, count: int, run_count: int, batch: bool) -> None:
    """Run the algorithm benchmarking.

    Args:
        definition (Definition): The algorithm definition.
        dataset_name (str): The name of the dataset.
        count (int): The number of results to return.
        run_count (int): The number of runs.
        batch (bool): If true, runs in batch mode.
    """
    algo = instantiate_algorithm(definition)
    assert not definition.query_argument_groups or hasattr(
        algo, "set_query_arguments"
    ), f"""\
error: query argument groups have been specified for {definition.module}.{definition.constructor}({definition.arguments}), but the \
algorithm instantiated from it does not implement the set_query_arguments \
function"""

    X_train, X_test, distance = load_and_transform_dataset_lazy(dataset_name)

    try:
        if hasattr(algo, "supports_prepared_queries"):
            algo.supports_prepared_queries()

        build_time, index_size = build_index(algo, X_train)

        query_argument_groups = definition.query_argument_groups or [[]]  # Ensure at least one iteration

        for pos, query_arguments in enumerate(query_argument_groups, 1):
            print(f"Running query argument group {pos} of {len(query_argument_groups)}...")
            if query_arguments:
                algo.set_query_arguments(*query_arguments)
            
            descriptor, results = run_individual_query(algo, X_train, X_test, distance, count, run_count, batch)

            descriptor.update({
                "build_time": build_time,
                "index_size": index_size,
                "algo": definition.algorithm,
                "dataset": dataset_name
            })

            store_results(dataset_name, count, definition, query_arguments, descriptor, results, batch)
    finally:
        algo.done()

def run_from_cmdline():
    """Calls the function `run` using arguments from the command line. See `ArgumentParser` for 
    arguments, all run it with `--help`.
    """
    parser = argparse.ArgumentParser(
        """

            NOTICE: You probably want to run.py rather than this script.

"""
    )
    parser.add_argument("--dataset", choices=DATASETS.keys(), help="Dataset to benchmark on.", required=True)
    parser.add_argument("--algorithm", help="Name of algorithm for saving the results.", required=True)
    parser.add_argument(
        "--module", help='Python module containing algorithm. E.g. "ann_benchmarks.algorithms.annoy"', required=True
    )
    parser.add_argument("--constructor", help='Constructer to load from modulel. E.g. "Annoy"', required=True)
    parser.add_argument(
        "--count", help="K: Number of nearest neighbours for the algorithm to return.", required=True, type=int
    )
    parser.add_argument(
        "--runs",
        help="Number of times to run the algorihm. Will use the fastest run-time over the bunch.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--batch",
        help='If flag included, algorithms will be run in batch mode, rather than "individual query" mode.',
        action="store_true",
    )
    parser.add_argument("build", help='JSON of arguments to pass to the constructor. E.g. ["angular", 100]')
    parser.add_argument("queries", help="JSON of arguments to pass to the queries. E.g. [100]", nargs="*", default=[])
    args = parser.parse_args()

    algo_args = json.loads(args.build)
    print(algo_args)
    query_args = [json.loads(q) for q in args.queries]

    definition = Definition(
        algorithm=args.algorithm,
        docker_tag=None,  # not needed
        module=args.module,
        constructor=args.constructor,
        arguments=algo_args,
        query_argument_groups=query_args,
        disabled=False,
    )
    run(definition, args.dataset, args.count, args.runs, args.batch)


def run_docker(
    definition: Definition,
    dataset: str,
    count: int,
    runs: int,
    timeout: int,
    batch: bool,
    cpu_limit: str,
    mem_limit: Optional[int] = None
) -> None:
    """Runs `run_from_cmdline` within a Docker container with specified parameters and logs the output.

    See `run_from_cmdline` for details on the args.
    """
    cmd = [
        "--dataset",
        dataset,
        "--algorithm",
        definition.algorithm,
        "--module",
        definition.module,
        "--constructor",
        definition.constructor,
        "--runs",
        str(runs),
        "--count",
        str(count),
    ]
    if batch:
        cmd += ["--batch"]
    cmd.append(json.dumps(definition.arguments))
    cmd += [json.dumps(qag) for qag in definition.query_argument_groups]

    client = docker.from_env()
    if mem_limit is None:
        mem_limit = psutil.virtual_memory().available

    container = client.containers.run(
        definition.docker_tag,
        cmd,
        volumes={
            os.path.abspath("/var/run/docker.sock"): {"bind": "/var/run/docker.sock", "mode": "rw"},
            os.path.abspath("ann_benchmarks"): {"bind": "/home/app/ann_benchmarks", "mode": "ro"},
            os.path.abspath("data"): {"bind": "/home/app/data", "mode": "ro"},
            os.path.abspath("results"): {"bind": "/home/app/results", "mode": "rw"},
        },
        network_mode="host",
        cpuset_cpus=cpu_limit,
        mem_limit=mem_limit,
        detach=True,
    )
    logger = logging.getLogger(f"annb.{container.short_id}")

    logger.info(
        "Created container %s: CPU limit %s, mem limit %s, timeout %d, command %s"
        % (container.short_id, cpu_limit, mem_limit, timeout, cmd)
    )

    def stream_logs():
        for line in container.logs(stream=True):
            logger.info(colors.color(line.decode().rstrip(), fg="blue"))

    t = threading.Thread(target=stream_logs, daemon=True)
    t.start()

    try:
        return_value = container.wait(timeout=timeout)
        _handle_container_return_value(return_value, container, logger)
    except Exception as e:
        logger.error("Container.wait for container %s failed with exception", container.short_id)
        logger.error(str(e))
    finally:
        logger.info("Removing container")
        container.remove(force=True)


def _handle_container_return_value(
    return_value: Union[Dict[str, Union[int, str]], int],
    container: docker.models.containers.Container,
    logger: logging.Logger
) -> None:
    """Handles the return value of a Docker container and outputs error and stdout messages (with colour).

    Args:
        return_value (Union[Dict[str, Union[int, str]], int]): The return value of the container.
        container (docker.models.containers.Container): The Docker container.
        logger (logging.Logger): The logger instance.
    """

    base_msg = f"Child process for container {container.short_id} "
    msg = base_msg + "returned exit code {}"

    if isinstance(return_value, dict):  # The return value from container.wait changes from int to dict in docker 3.0.0
        error_msg = return_value.get("Error", "")
        exit_code = return_value["StatusCode"]
        msg = msg.format(f"{exit_code} with message {error_msg}")
    else:
        exit_code = return_value
        msg = msg.format(exit_code)

    if exit_code not in [0, None]:
        for line in container.logs(stream=True):
            logger.error(colors.color(line.decode(), fg="red"))
        logger.error(msg)
    else:
        logger.info(msg)
