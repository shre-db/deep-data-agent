"""Tests for the sandbox chart helpers (no Streamlit runtime needed)."""

import pandas as pd
import plotly.io as pio

from app.agent_tools.charts import LIGHT_CATEGORICAL, bar, line, scatter


def test_bar_saves_parseable_figure(tmp_path):
    df = pd.DataFrame({"channel": ["A", "B"], "revenue": [10, 20]})
    bar(df, x="channel", y="revenue", title="Revenue", path=tmp_path / "rev.json")
    fig = pio.from_json((tmp_path / "rev.json").read_text())
    assert len(fig.data) == 1
    assert fig.layout.title.text == "Revenue"
    assert fig.data[0].marker.color == LIGHT_CATEGORICAL[0]
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"


def test_line_and_scatter_save(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    line(df, x="x", y="y", title="Trend", path=tmp_path / "trend.json")
    scatter(df, x="x", y="y", title="Points", path=tmp_path / "pts.json")
    trend = pio.from_json((tmp_path / "trend.json").read_text())
    pts = pio.from_json((tmp_path / "pts.json").read_text())
    assert trend.data[0].line.width == 2
    assert pts.data[0].marker.size == 9
    assert pts.layout.title.text == "Points"


def test_multi_series_uses_fixed_categorical_order(tmp_path):
    df = pd.DataFrame({"g": ["a", "b", "c"], "x": [1, 2, 3], "y": [2, 3, 4]})
    line(df, x="x", y="y", color="g", path=tmp_path / "multi.json")
    fig = pio.from_json((tmp_path / "multi.json").read_text())
    assert [t.line.color for t in fig.data] == LIGHT_CATEGORICAL[:3]


def test_style_fits_axis_titles_in_frame():
    """px sets axis titles to column names by default; margins must make
    room for them and automargin must handle tick labels."""
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    fig = bar(df, x="x", y="y", title="T")
    assert fig.layout.xaxis.automargin is True
    assert fig.layout.yaxis.automargin is True
    assert fig.layout.margin.b == 52  # xaxis title present
    assert fig.layout.margin.l == 64  # yaxis title present
    assert fig.layout.margin.t == 44  # chart title, no legend


def test_style_moves_legend_horizontal_when_present():
    df = pd.DataFrame({"g": ["a", "b"], "x": [1, 2], "y": [3, 4]})
    fig = bar(df, x="x", y="y", color="g", title="T")
    assert fig.layout.legend.orientation == "h"
    assert fig.layout.margin.t == 44 + 32  # title + legend room
