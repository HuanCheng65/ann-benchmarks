from pathlib import Path
import importlib.util


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "plot_param_slice.py"
SPEC = importlib.util.spec_from_file_location("plot_param_slice", SCRIPT_PATH)
plot_param_slice = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plot_param_slice)


def test_numeric_x_values_use_numeric_positions():
    rows = [
        {"x_value_raw": "64", "y_value": 1.0},
        {"x_value_raw": "256", "y_value": 2.0},
        {"x_value_raw": "1024", "y_value": 3.0},
    ]

    x_values, x_tick_positions, x_tick_labels, categorical = plot_param_slice.build_x_axis(rows)

    assert x_values == [64.0, 256.0, 1024.0]
    assert x_tick_positions is None
    assert x_tick_labels is None
    assert categorical is False


def test_string_x_values_fall_back_to_categorical_axis():
    rows = [
        {"x_value_raw": "small", "y_value": 1.0},
        {"x_value_raw": "large", "y_value": 2.0},
    ]

    x_values, x_tick_positions, x_tick_labels, categorical = plot_param_slice.build_x_axis(rows)

    assert x_values == [0, 1]
    assert x_tick_positions == [0, 1]
    assert x_tick_labels == ["small", "large"]
    assert categorical is True
