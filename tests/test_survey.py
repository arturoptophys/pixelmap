"""Tests for pixelmap.utils.survey — survey .txt parsing and probe-match."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pixelmap.constants import WIRING_FILE_MAP
from pixelmap.types import Electrode
from pixelmap.utils import survey

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def positions_df_1_0(wiring_maps_dir):
    return pd.read_csv(wiring_maps_dir / WIRING_FILE_MAP["1.0"][0])


@pytest.fixture
def positions_df_2_0_1shank(wiring_maps_dir):
    return pd.read_csv(wiring_maps_dir / WIRING_FILE_MAP["2.0-1shank"][0])


@pytest.fixture
def positions_df_2_0_4shanks(wiring_maps_dir):
    return pd.read_csv(wiring_maps_dir / WIRING_FILE_MAP["2.0-4shanks"][0])


def _survey_content_from_positions(positions_df: pd.DataFrame, val_of) -> str:
    lines = ["Shank\tXum\tZum\tVal"]
    for shank, _elec, x, y in positions_df[["shank", "electrode", "x", "y"]].itertuples(index=False):
        lines.append(f"{int(shank)}\t{int(x)}\t{int(y)}\t{val_of(shank, x, y)}")
    return "\n".join(lines) + "\n"


def test_parse_survey_file_happy_path():
    content = "Shank\tXum\tZum\tVal\n0\t27\t0\t1.5\n0\t59\t0\t2.0\n"
    df = survey.parse_survey_file(content)
    assert list(df.columns) == ["shank", "xum", "zum", "val"]
    assert len(df) == 2
    assert df.iloc[1]["val"] == pytest.approx(2.0)


def test_parse_survey_file_4shank_export():
    df = survey.parse_survey_file((FIXTURES / "survey_2.0-4shanks.txt").read_text())
    assert set(df.columns) == {"shank", "xum", "zum", "val"}
    assert df["shank"].min() >= 0
    assert df["shank"].max() <= 3
    assert set(df["shank"].unique()) == {0, 1, 2, 3}


def test_parse_survey_file_1_0_export():
    df = survey.parse_survey_file((FIXTURES / "survey_1.0.txt").read_text())
    assert set(df.columns) == {"shank", "xum", "zum", "val"}
    assert set(df["shank"].unique()) == {0}


def test_parse_survey_file_2_0_1shank_export():
    df = survey.parse_survey_file((FIXTURES / "survey_2.0-1shank.txt").read_text())
    assert set(df.columns) == {"shank", "xum", "zum", "val"}
    assert set(df["shank"].unique()) == {0}


def test_validate_probe_match_1_0_export(positions_df_1_0):
    df = survey.parse_survey_file((FIXTURES / "survey_1.0.txt").read_text())
    values, n_unmatched = survey.validate_probe_match(df, positions_df_1_0)
    assert n_unmatched == 0
    assert len(values) == len(positions_df_1_0)


def test_validate_probe_match_2_0_1shank_export(positions_df_2_0_1shank):
    df = survey.parse_survey_file((FIXTURES / "survey_2.0-1shank.txt").read_text())
    values, n_unmatched = survey.validate_probe_match(df, positions_df_2_0_1shank)
    assert n_unmatched == 0
    assert len(values) == len(positions_df_2_0_1shank)


def test_parse_survey_file_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        survey.parse_survey_file("foo\tbar\n1\t2\n")


def test_validate_probe_match_success(positions_df_1_0):
    content = _survey_content_from_positions(positions_df_1_0, lambda s, x, y: float(y))
    df = survey.parse_survey_file(content)
    values, n_unmatched = survey.validate_probe_match(df, positions_df_1_0)
    assert n_unmatched == 0
    assert len(values) == len(positions_df_1_0)
    # Spot-check: Val encodes y, so Electrode(0, 0) (at y=0) should have val 0
    assert values[Electrode(0, 0)] == pytest.approx(0.0)


def test_validate_probe_match_rejects_wrong_probe(positions_df_2_0_4shanks, positions_df_1_0):
    # Build survey from 4-shank positions, but validate against 1.0 positions
    content = _survey_content_from_positions(positions_df_2_0_4shanks, lambda s, x, y: 1.0)
    df = survey.parse_survey_file(content)
    with pytest.raises(ValueError, match="different probe type|No survey rows matched"):
        survey.validate_probe_match(df, positions_df_1_0)


def test_match_handles_constant_offset(positions_df_2_0_4shanks):
    """SpikeGLX ships shank-local coords; positions CSV uses global x.
    Survey with an arbitrary per-shank offset should still match within tolerance."""
    rows = []
    for shank, _elec, x, y in positions_df_2_0_4shanks[["shank", "electrode", "x", "y"]].itertuples(index=False):
        # Simulate SpikeGLX: subtract shank's min-x (shank-local) and add 27µm offset
        shank_min_x = positions_df_2_0_4shanks[positions_df_2_0_4shanks["shank"] == shank]["x"].min()
        local_x = int(x) - int(shank_min_x) + 27
        rows.append(f"{int(shank)}\t{local_x}\t{int(y)}\t1.0")
    content = "Shank\tXum\tZum\tVal\n" + "\n".join(rows) + "\n"
    df = survey.parse_survey_file(content)
    values, n_unmatched = survey.validate_probe_match(df, positions_df_2_0_4shanks)
    assert n_unmatched == 0
    assert len(values) == len(positions_df_2_0_4shanks)


class TestDefaultSurveyRange:
    """Initial colormap bounds: robust enough that the overlay reads on sight."""

    def test_tails_are_clipped_off_a_heavy_tailed_survey(self):
        rng = np.random.default_rng(0)
        vals = list(rng.gamma(2.0, 1.0, size=500)) + [500.0, 900.0]
        vmin, vmax = survey.default_survey_range(vals)
        assert vmin > min(vals)
        assert vmax < max(vals) / 10  # the two outliers no longer set the scale

    def test_bulk_of_the_data_stays_inside_the_range(self):
        rng = np.random.default_rng(1)
        vals = rng.gamma(2.0, 1.0, size=1000)
        vmin, vmax = survey.default_survey_range(vals)
        inside = ((vals >= vmin) & (vals <= vmax)).mean()
        assert inside > 0.9

    def test_percentiles_are_configurable(self):
        vals = list(range(101))
        assert survey.default_survey_range(vals, percentiles=(0.0, 100.0)) == (0.0, 100.0)
        assert survey.default_survey_range(vals, percentiles=(10.0, 90.0)) == (10.0, 90.0)

    def test_constant_survey_gets_a_non_degenerate_range(self):
        vmin, vmax = survey.default_survey_range([4.2] * 50)
        assert vmax > vmin

    def test_nearly_constant_survey_falls_back_to_the_full_range(self):
        # Percentiles collapse (99% of values identical) but min/max still differ.
        vals = [1.0] * 500 + [9.0]
        vmin, vmax = survey.default_survey_range(vals)
        assert (vmin, vmax) == (1.0, 9.0)

    def test_non_finite_values_are_ignored(self):
        vmin, vmax = survey.default_survey_range(
            [float("nan"), float("inf"), -float("inf")] + list(range(101))
        )
        assert np.isfinite(vmin) and np.isfinite(vmax)
        assert 0.0 <= vmin < vmax <= 100.0

    def test_empty_survey_gets_a_safe_default(self):
        assert survey.default_survey_range([]) == (0.0, 1.0)
        assert survey.default_survey_range([float("nan")]) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Open Ephys (Neuropix-PXI plugin) survey JSON
# ---------------------------------------------------------------------------


def _oe_electrodes(oe_type: str, surveyed_shanks=None, val_of=lambda s, x, y: float(y)):
    """Electrode list built with the plugin's Geometry.cpp formulas."""
    out = []
    if oe_type == "Neuropixels 1.0":
        n_per_shank, n_shanks = 960, 1
    elif oe_type == "Neuropixels 2.0 Single Shank":
        n_per_shank, n_shanks = 1280, 1
    else:
        n_per_shank, n_shanks = 1280, 4
    for i in range(n_per_shank * n_shanks):
        shank, local = divmod(i, n_per_shank)
        if oe_type == "Neuropixels 1.0":
            x = [27.0, 59.0, 11.0, 43.0][i % 4]
            y = (i - i % 2) * 10.0
        else:
            x = i % 2 * 32.0 + 27.0
            y = (local - local % 2) * 7.5
        surveyed = surveyed_shanks is None or shank in surveyed_shanks
        out.append({
            "global_index": i, "shank": shank, "column": i % 2, "row": local // 2,
            "bank": "A", "is_reference": False,
            "position_x_um": x, "position_y_um": y,
            "was_surveyed": surveyed,
            "peak_to_peak": val_of(shank, x, y) if surveyed else 0.0,
        })
    return out


def _oe_survey_json(*probes) -> str:
    """probes: (name, oe_type, electrodes) tuples."""
    return json.dumps({
        "generated_at": "2026-01-01T00:00:00.000Z",
        "seconds_per_configuration": 2.0,
        "probe_count": len(probes),
        "probes": [
            {"name": name, "type": oe_type, "shank_count": 1, "sample_rate": 30000.0,
             "serial_number": "123", "banks_surveyed": ["A"], "shanks_surveyed": [0],
             "electrodes": electrodes}
            for name, oe_type, electrodes in probes
        ],
    })


@pytest.mark.parametrize("oe_type, probe_type", [
    ("Neuropixels 1.0", "1.0"),
    ("Neuropixels 2.0 Single Shank", "2.0-1shank"),
    ("Neuropixels 2.0 Multi Shank", "2.0-4shanks"),
])
def test_openephys_json_matches_probe(wiring_maps_dir, oe_type, probe_type):
    positions_df = pd.read_csv(wiring_maps_dir / WIRING_FILE_MAP[probe_type][0])
    content = _oe_survey_json(("ProbeA", oe_type, _oe_electrodes(oe_type)))
    assert survey.is_openephys_survey(content)

    (probe,) = survey.parse_openephys_survey_json(content)
    assert probe.probe_type == probe_type
    assert list(probe.survey_df.columns) == list(survey.SURVEY_COLUMNS)

    values, n_unmatched = survey.validate_probe_match(probe.survey_df, positions_df)
    assert n_unmatched == 0
    assert len(values) == len(positions_df)
    # val encodes y, so every electrode must land on a site at the same depth
    y_by_elec = {
        Electrode(int(s), int(e)): y
        for s, e, y in positions_df[["shank", "electrode", "y"]].itertuples(index=False)
    }
    for elec, val in values.items():
        assert val == pytest.approx(y_by_elec[elec])


def test_openephys_json_drops_unsurveyed_and_reference():
    electrodes = _oe_electrodes("Neuropixels 2.0 Multi Shank", surveyed_shanks={1})
    electrodes[1280]["is_reference"] = True  # first site on shank 1
    (probe,) = survey.parse_openephys_survey_json(
        _oe_survey_json(("P", "Neuropixels 2.0 Multi Shank", electrodes))
    )
    assert set(probe.survey_df["shank"]) == {1}
    assert len(probe.survey_df) == 1279


def test_openephys_json_multi_probe_and_unsupported_type():
    content = _oe_survey_json(
        ("P1", "Neuropixels 1.0", _oe_electrodes("Neuropixels 1.0")),
        ("P2", "Neuropixels UHD (Fixed)", _oe_electrodes("Neuropixels 1.0")),
        ("P3", "Neuropixels 2.0 Single Shank",
         _oe_electrodes("Neuropixels 2.0 Single Shank", surveyed_shanks=set())),
    )
    probes = survey.parse_openephys_survey_json(content)
    # P3 has no surveyed electrodes and is skipped entirely
    assert [p.name for p in probes] == ["P1", "P2"]
    assert probes[1].probe_type is None
    assert "SN 123" in probes[0].label


@pytest.mark.parametrize("content, match", [
    ("{not json", "Could not parse survey file as JSON"),
    ('{"probes": []}', "no 'probes' list"),
    ('{"probes": [{"name": "x", "electrodes": [{"shank": 0}]}]}', "Malformed electrode entry"),
])
def test_openephys_json_errors(content, match):
    with pytest.raises(ValueError, match=match):
        survey.parse_openephys_survey_json(content)


def test_spikeglx_txt_is_not_sniffed_as_json():
    assert not survey.is_openephys_survey((FIXTURES / "survey_1.0.txt").read_text())
