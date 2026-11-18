########################################
## SpikeGLX / Open Ephys survey I/O   ##
########################################

import json
from dataclasses import dataclass
from io import StringIO

import numpy as np
import pandas as pd

from pixelmap.types import Electrode

SURVEY_COLUMNS = ("shank", "xum", "zum", "val")

# Tolerance for matching a survey (Xum, Zum) to a probe electrode.
# SpikeGLX uses shank-local coordinates that can differ from our
# positions CSVs by a constant per-probe offset; the tolerance absorbs
# that while still rejecting surveys from a genuinely different probe.
DEFAULT_MATCH_TOLERANCE_UM = 35.0

# Percentiles used to seed the colormap when a survey is loaded. Survey
# metrics (spike rate, amplitude) are heavy-tailed: a handful of very active
# contacts stretch a raw min→max range so far that every other contact lands
# in the bottom of the colormap and the overlay reads as uniformly dark.
# Clipping the tails gives a heatmap with visible structure on first render.
DEFAULT_RANGE_PERCENTILES = (2.0, 98.0)


def default_survey_range(
    values,
    percentiles: tuple[float, float] = DEFAULT_RANGE_PERCENTILES,
) -> tuple[float, float]:
    """Pick sensible initial ``(vmin, vmax)`` colormap bounds for a survey.

    Uses robust percentiles rather than the raw min/max (see
    :data:`DEFAULT_RANGE_PERCENTILES`), falling back to the full range, and
    finally to a unit-wide range, whenever the percentiles collapse to a
    single value. The result is always a non-degenerate ``vmin < vmax``.

    Args:
        values: iterable of survey values (NaNs and infinities are ignored).
        percentiles: ``(low, high)`` percentiles to clip the tails at.

    Returns:
        ``(vmin, vmax)`` with ``vmax > vmin``.
    """
    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0

    low_pct, high_pct = percentiles
    vmin, vmax = (float(v) for v in np.percentile(vals, [low_pct, high_pct]))
    if vmax <= vmin:  # tails clipped away all the contrast — use the full range
        vmin, vmax = float(vals.min()), float(vals.max())
    if vmax <= vmin:  # genuinely constant survey — avoid a degenerate colormap
        vmax = vmin + 1.0
    return vmin, vmax


def parse_survey_file(content: str) -> pd.DataFrame:
    """
    Parse a SpikeGLX-exported survey .txt file.

    The file is tab-separated with a header line: Shank, Xum, Zum, Val.
    Each subsequent row corresponds to one probe contact and its measured
    value (spike rate, amplitude, LFP power, etc).

    Returns a DataFrame with lowercase columns (shank, xum, zum, val).
    Raises ValueError if the header or dtypes are unexpected.
    """
    try:
        df = pd.read_csv(StringIO(content), sep="\t")
    except Exception as e:
        raise ValueError(f"Could not parse survey file as TSV: {e}") from e

    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in SURVEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Survey file is missing required columns: {missing}. "
            f"Expected header: Shank\\tXum\\tZum\\tVal."
        )

    df = (
        df[list(SURVEY_COLUMNS)]
        .apply(pd.to_numeric, errors="raise")
        .astype({"shank": int, "xum": int, "zum": int, "val": float})
    )
    return df


# Open Ephys Neuropix-PXI plugin probe type strings (``probeTypeToString``)
# mapped to PixelMap probe types. Types absent here are not supported by
# PixelMap and are rejected when loading a survey.
OPENEPHYS_PROBE_TYPE_MAP = {
    "Neuropixels 1.0": "1.0",
    "Neuropixels 2.0 Single Shank": "2.0-1shank",
    "Neuropixels 2.0 Multi Shank": "2.0-4shanks",
    "Neuropixels 2.0 Quad Base": "QuadBase",
}


@dataclass
class OpenEphysSurveyProbe:
    """One probe's results from an Open Ephys survey JSON export."""

    name: str
    oe_type: str
    probe_type: str | None  # PixelMap probe type, or None if unsupported
    serial_number: str | None
    survey_df: pd.DataFrame  # columns SURVEY_COLUMNS, surveyed contacts only

    @property
    def label(self) -> str:
        serial = f", SN {self.serial_number}" if self.serial_number else ""
        return f"{self.name} ({self.oe_type}{serial})"


def is_openephys_survey(content: str) -> bool:
    """Cheap sniff: does this look like a JSON document rather than a TSV?"""
    return content.lstrip().startswith("{")


def parse_openephys_survey_json(content: str) -> list[OpenEphysSurveyProbe]:
    """
    Parse an Open Ephys survey JSON (Neuropix-PXI plugin, Survey tab → Save).

    The file holds a ``probes`` list; each probe has ``name``, ``type`` and an
    ``electrodes`` list whose entries carry ``shank``, ``position_x_um``,
    ``position_y_um`` (shank-local, 0 at the lowest site), ``peak_to_peak``,
    ``was_surveyed`` and ``is_reference``. Unsurveyed electrodes report 0 and
    reference sites carry no signal, so both are dropped.

    Each probe's surveyed contacts are converted to the same
    (shank, xum, zum, val) DataFrame that :func:`parse_survey_file` produces,
    so downstream matching is format-agnostic.

    Raises ValueError if the JSON is malformed or contains no surveyed probes.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse survey file as JSON: {e}") from e

    raw_probes = data.get("probes") if isinstance(data, dict) else None
    if not isinstance(raw_probes, list) or not raw_probes:
        raise ValueError(
            "JSON file has no 'probes' list — expected an Open Ephys survey export."
        )

    probes: list[OpenEphysSurveyProbe] = []
    for i, raw in enumerate(raw_probes):
        electrodes = raw.get("electrodes") if isinstance(raw, dict) else None
        if not isinstance(electrodes, list) or not electrodes:
            raise ValueError(f"Probe #{i} in survey JSON has no 'electrodes' list.")
        rows = []
        for e in electrodes:
            try:
                if not e.get("was_surveyed", True) or e.get("is_reference", False):
                    continue
                rows.append((
                    int(e["shank"]),
                    float(e["position_x_um"]),
                    float(e["position_y_um"]),
                    float(e["peak_to_peak"]),
                ))
            except (AttributeError, KeyError, TypeError, ValueError) as err:
                raise ValueError(
                    f"Malformed electrode entry in probe #{i} of survey JSON: {err}"
                ) from err
        if not rows:
            continue
        oe_type = str(raw.get("type", "Unknown"))
        serial = raw.get("serial_number")
        probes.append(OpenEphysSurveyProbe(
            name=str(raw.get("name", f"Probe {i}")),
            oe_type=oe_type,
            probe_type=OPENEPHYS_PROBE_TYPE_MAP.get(oe_type),
            serial_number=str(serial) if serial else None,
            survey_df=pd.DataFrame(rows, columns=list(SURVEY_COLUMNS)),
        ))

    if not probes:
        raise ValueError("Survey JSON contains no surveyed electrodes.")
    return probes


def _shank_local_positions(positions_df: pd.DataFrame) -> dict[int, np.ndarray]:
    """
    Return {shank_id: array of (electrode_id, local_x, y)} per shank.

    Positions CSVs for multi-shank probes store x in *global* coordinates
    (offset by shank index × shank spacing). SpikeGLX Xum is exported in
    *shank-local* coordinates. We normalise by subtracting the per-shank
    minimum x so lookups work regardless of convention.
    """
    out: dict[int, np.ndarray] = {}
    for shank_id, group in positions_df.groupby("shank"):
        x = group["x"].to_numpy(dtype=float)
        y = group["y"].to_numpy(dtype=float)
        local_x = x - x.min()
        elec = group["electrode"].to_numpy(dtype=int)
        out[int(shank_id)] = np.column_stack([elec, local_x, y])
    return out


def match_survey_to_electrodes(
    survey_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    tolerance_um: float = DEFAULT_MATCH_TOLERANCE_UM,
) -> tuple[dict[Electrode, float], int]:
    """
    Map each survey row to its nearest electrode on the same shank.

    A row is accepted if the nearest electrode is within ``tolerance_um``
    microns (Euclidean, after shank-local normalisation). Any row whose
    shank is absent from ``positions_df`` or whose nearest electrode is
    beyond the tolerance is counted as unmatched.

    Returns (values_by_electrode, n_unmatched).
    """
    shank_tables = _shank_local_positions(positions_df)
    survey_shanks = survey_df["shank"].to_numpy(dtype=int)
    survey_x = survey_df["xum"].to_numpy(dtype=float)
    survey_z = survey_df["zum"].to_numpy(dtype=float)
    survey_v = survey_df["val"].to_numpy(dtype=float)

    # Normalise survey x per-shank the same way the probe positions are
    # normalised, so the two coordinate systems line up regardless of
    # absolute offset.
    local_survey_x = survey_x.copy()
    for shank_id in np.unique(survey_shanks):
        mask = survey_shanks == shank_id
        local_survey_x[mask] -= survey_x[mask].min()

    values: dict[Electrode, float] = {}
    n_unmatched = 0
    tol2 = tolerance_um ** 2
    for i in range(len(survey_df)):
        shank_id = int(survey_shanks[i])
        table = shank_tables.get(shank_id)
        if table is None:
            n_unmatched += 1
            continue
        dx = table[:, 1] - local_survey_x[i]
        dy = table[:, 2] - survey_z[i]
        d2 = dx * dx + dy * dy
        j = int(np.argmin(d2))
        if d2[j] > tol2:
            n_unmatched += 1
            continue
        electrode_id = int(table[j, 0])
        values[Electrode(shank_id, electrode_id)] = float(survey_v[i])
    return values, n_unmatched


def validate_probe_match(
    survey_df: pd.DataFrame,
    positions_df: pd.DataFrame,
    tolerance_um: float = DEFAULT_MATCH_TOLERANCE_UM,
) -> tuple[dict[Electrode, float], int]:
    """
    Confirm the survey file matches the currently loaded probe type.

    Returns (values_by_electrode, n_unmatched) on success.
    Raises ValueError if no survey rows map to the probe (strong signal
    of a probe-type mismatch) or if the match rate is implausibly low.
    """
    values, n_unmatched = match_survey_to_electrodes(survey_df, positions_df, tolerance_um)
    total = len(survey_df)
    if not values:
        raise ValueError(
            "No survey rows matched the currently selected probe. "
            "Check that the survey was exported from the same probe type "
            "that is selected in the GUI."
        )
    # If fewer than half the rows mapped, the shapes don't align — almost
    # certainly a probe-type mismatch even if a few rows happen to fit.
    if total and len(values) / total < 0.5:
        raise ValueError(
            f"Only {len(values)}/{total} survey rows matched the selected probe — "
            "this survey was likely exported from a different probe type."
        )
    return values, n_unmatched
