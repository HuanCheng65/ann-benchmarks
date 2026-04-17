#!/usr/bin/env python3
import argparse
import math
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import h5py
import matplotlib as mpl

mpl.use("Agg")  # noqa
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from ann_benchmarks.datasets import get_dataset
from ann_benchmarks.plotting.metrics import all_metrics as metrics


PARAM_PAIR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^,()]+)")


def resolve_results_dir(args):
    if args.results_dir:
        return Path(args.results_dir)
    if not args.dataset or args.count is None or not args.algorithm:
        raise SystemExit(
            "Use --results-dir or provide --dataset, --count, and --algorithm."
        )
    algo_dir = args.algorithm + ("-batch" if args.batch else "")
    return Path("results") / args.dataset / str(args.count) / algo_dir


def parse_filters(filters):
    parsed = []
    for item in filters:
        if "=" not in item:
            raise SystemExit(f"Invalid --where '{item}', expected key=value.")
        key, value = item.split("=", 1)
        parsed.append((key.strip(), value.strip()))
    return parsed


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_value(value):
    if isinstance(value, str):
        return value.strip()
    return str(value)


def coerce_sort_value(value):
    text = normalize_value(value)
    text = text.rstrip("%")
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def metric_value(metric_name, true_distances, run_distances, metrics_cache, times, properties):
    return metrics[metric_name]["function"](true_distances, run_distances, metrics_cache, times, properties)


class ReadOnlyMetricsCache:
    def __init__(self):
        self.attrs = {}
        self.groups = {}

    def __contains__(self, key):
        return key in self.groups

    def __getitem__(self, key):
        return self.groups[key]

    def create_group(self, key):
        group = ReadOnlyMetricsCache()
        self.groups[key] = group
        return group


def load_rows(results_dir, dataset_name, filters, x_param, y_metric):
    dataset_file, _ = get_dataset(dataset_name)
    true_distances = np.array(dataset_file["distances"])
    rows = []

    for path in sorted(results_dir.glob("*.hdf5")):
        with h5py.File(path, "r") as run:
            properties = {k: decode_attr(v) for k, v in run.attrs.items()}
            name = normalize_value(properties.get("name", ""))
            parsed = dict(PARAM_PAIR_RE.findall(name))

            if x_param not in parsed:
                continue

            matched = True
            for key, expected in filters:
                actual = parsed.get(key)
                if actual is None or normalize_value(actual) != expected:
                    matched = False
                    break
            if not matched:
                continue

            metrics_cache = ReadOnlyMetricsCache()
            run_distances = np.array(run["distances"])
            times = np.array(run["times"])
            y_value = metric_value(y_metric, true_distances, run_distances, metrics_cache, times, properties)

            rows.append(
                {
                    "file": str(path),
                    "name": name,
                    "x_param": x_param,
                    "x_value_raw": parsed[x_param],
                    "x_sort": coerce_sort_value(parsed[x_param]),
                    "y_metric": y_metric,
                    "y_value": float(y_value),
                    "params": parsed,
                }
            )

    rows.sort(key=lambda row: row["x_sort"])
    return rows


def choose_default_output(args, results_dir):
    if args.output:
        return Path(args.output)
    dataset_part = args.dataset or results_dir.parent.parent.name
    algorithm_part = args.algorithm or results_dir.name
    filter_part = "_".join(f"{k}_{v}" for k, v in parse_filters(args.where)) or "all"
    filename = f"{dataset_part}-{algorithm_part}-{args.x_param}-vs-{args.y_metric}-{filter_part}.png"
    return Path("results") / filename


def maybe_log_scale(values, requested):
    if requested != "auto":
        return requested
    finite_positive = [v for v in values if v > 0 and math.isfinite(v)]
    if len(finite_positive) < 2:
        return "linear"
    span = max(finite_positive) / min(finite_positive)
    return "log" if span >= 20 else "linear"


def build_x_axis(rows):
    numeric_values = []
    for row in rows:
        x_sort = row.get("x_sort")
        if x_sort is None:
            x_sort = coerce_sort_value(row["x_value_raw"])
        sort_kind, sort_value = x_sort
        if sort_kind != 0:
            numeric_values = None
            break
        numeric_values.append(float(sort_value))

    if numeric_values is not None:
        return numeric_values, None, None, False

    x_labels = [normalize_value(row["x_value_raw"]) for row in rows]
    x_positions = list(range(len(rows)))
    return x_positions, x_positions, x_labels, True


def render_plot(rows, output_path, title, x_label, y_label, x_scale, y_scale):
    x_values, x_tick_positions, x_tick_labels, categorical = build_x_axis(rows)
    y_values = [row["y_value"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker="o", linewidth=2)
    if x_tick_positions is not None:
        plt.xticks(x_tick_positions, x_tick_labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    ax = plt.gca()
    ax.set_yscale(y_scale)
    if not categorical:
        ax.set_xscale(x_scale)
        numeric_ticks = [float(value) for value in x_values]
        ax.set_xticks(numeric_ticks)
        ax.set_xticklabels([normalize_value(row["x_value_raw"]) for row in rows])

        if x_scale == "log":
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            xmin = min(numeric_ticks) / 1.25
            xmax = max(numeric_ticks) * 1.25
        else:
            span = max(numeric_ticks) - min(numeric_ticks)
            margin = span * 0.08 if span else max(abs(numeric_ticks[0]) * 0.08, 1.0)
            xmin = min(numeric_ticks) - margin
            xmax = max(numeric_ticks) + margin
        ax.set_xlim(xmin, xmax)

    ax.grid(True, which="major", alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def print_table(rows):
    print("x_value\ty_value\tname")
    for row in rows:
        print(f"{row['x_value_raw']}\t{row['y_value']:.6f}\t{row['name']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir")
    parser.add_argument("--dataset")
    parser.add_argument("--count", type=int)
    parser.add_argument("--algorithm")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--x-param", required=True)
    parser.add_argument("--y-metric", choices=metrics.keys(), required=True)
    parser.add_argument("--where", action="append", default=[])
    parser.add_argument("--output")
    parser.add_argument("--title")
    parser.add_argument("--x-label")
    parser.add_argument("--y-label")
    parser.add_argument("--x-scale", choices=["auto", "linear", "log"], default="auto")
    parser.add_argument("--y-scale", choices=["auto", "linear", "log"], default="auto")
    args = parser.parse_args()

    results_dir = resolve_results_dir(args)
    if not results_dir.exists():
        raise SystemExit(f"Results directory does not exist: {results_dir}")

    dataset_name = args.dataset
    if not dataset_name:
        try:
            dataset_name = results_dir.parent.parent.name
        except IndexError as exc:
            raise SystemExit("Could not infer dataset name from results directory.") from exc

    filters = parse_filters(args.where)
    rows = load_rows(results_dir, dataset_name, filters, args.x_param, args.y_metric)
    if not rows:
        raise SystemExit("No matching result files found.")

    output_path = choose_default_output(args, results_dir)
    title = args.title or f"{args.y_metric} vs {args.x_param}"
    x_label = args.x_label or args.x_param
    y_label = args.y_label or metrics[args.y_metric]["description"]
    x_scale = maybe_log_scale([row["x_sort"][1] for row in rows if row["x_sort"][0] == 0], args.x_scale)
    y_scale = maybe_log_scale([row["y_value"] for row in rows], args.y_scale)

    print_table(rows)
    render_plot(rows, output_path, title, x_label, y_label, x_scale, y_scale)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
