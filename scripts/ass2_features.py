"""Pre-visit static and strict-date history features for Assessment 2.

build_features returns X, nullable y and ID/date/group metadata in input row order.
For validation/prediction pass an explicit history and batch_start.
The training pipeline handles imputation, encoding, scaling and model fitting.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import RAW_TO_CLEAN, identifier, tidy, sha256_file, KNOWN_RESULTS, CONFIG

ROOT = Path(__file__).resolve().parents[1]
BASE_CATEGORICAL_FEATURES = [
    "facility_type", "risk", "inspection_type", "zip", "inspection_month", "inspection_weekday",
]
BASE_NUMERIC_FEATURES = ["latitude", "longitude"]
HISTORY_CATEGORICAL_FEATURES = ["previous_outcome"]
HISTORY_NUMERIC_FEATURES = ["prior_visits", "prior_judgements", "prior_failures", "prior_fail_rate"]


@dataclass
class FeatureSet:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame


def _text(values):
    return values.astype("string").fillna("").map(tidy)


def _dates(values):
    text = _text(values)
    iso = text.str.match(r"^\d{4}-\d{2}-\d{2}(?:[ T].*)?$")
    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")
    result.loc[iso] = pd.to_datetime(text.loc[iso].str[:10], format="%Y-%m-%d", errors="raise")
    result.loc[~iso] = pd.to_datetime(text.loc[~iso], format="%m/%d/%Y", errors="raise")
    if result.isna().any():
        raise ValueError("Inspection dates must not be missing.")
    return result.dt.normalize()


def normalize_inspections(frame: pd.DataFrame) -> pd.DataFrame:
    """Accept raw portal names or prepared names; also accept unlabelled future inputs."""
    data = frame.rename(columns=RAW_TO_CLEAN).copy().reset_index(drop=True)
    if data.columns.duplicated().any():
        raise ValueError("Ambiguous raw/prepared column names.")
    if not {"inspection_id", "inspection_date"}.issubset(data.columns):
        raise ValueError("Inspection ID and Inspection Date are required.")
    out = pd.DataFrame(index=data.index)
    out["inspection_id"] = _text(data["inspection_id"]).map(identifier)
    if out.inspection_id.eq("").any() or out.inspection_id.duplicated().any():
        raise ValueError("Inspection IDs must be valid and unique.")
    out["inspection_date"] = _dates(data["inspection_date"])
    text_fields = [
        "dba_name", "license_number", "address", "city", "state",
        "zip", "facility_type", "risk", "inspection_type",
    ]
    for field in text_fields:
        values = data[field] if field in data else pd.Series("", index=data.index)
        out[field] = _text(values)
    out["license_number"] = out.license_number.map(identifier)
    for field in ["facility_type", "risk", "inspection_type"]:
        out[field] = out[field].str.upper().replace("", "UNKNOWN")
    for field in ["address", "city", "state"]:
        out[field] = out[field].str.upper()
    out["zip"] = out.zip.str.replace(r"^(\d{5})\.0$", r"\1", regex=True).replace("", "UNKNOWN")
    for field, limit in [("latitude", 90), ("longitude", 180)]:
        values = data[field] if field in data else pd.Series(np.nan, index=data.index)
        numbers = pd.to_numeric(values, errors="coerce").astype(float)
        out[field] = numbers.where(np.isfinite(numbers) & numbers.between(-limit, limit))

    labels = pd.Series(pd.NA, index=data.index, dtype="Int8", name="label")
    if "results" in data:
        results = _text(data["results"])
        canonical = results.str.casefold().map(KNOWN_RESULTS)
        if (results.ne("") & canonical.isna()).any():
            raise ValueError("Unrecognised Results value.")
        labels = canonical.map(CONFIG["label_map"]).astype("Int8").rename("label")
    if "label" in data:
        values = data["label"]
        numeric = pd.to_numeric(values, errors="coerce")
        present = _text(values).ne("")
        if (present & (numeric.isna() | ~numeric.isin([0, 1]))).any():
            raise ValueError("Labels must be 0, 1 or missing.")
        supplied = numeric.astype("Int8").rename("label")
        if "results" in data and not supplied.equals(labels):
            raise ValueError("Results and supplied labels disagree.")
        labels = supplied
    out["label"] = labels
    return out


def establishment_keys(data: pd.DataFrame, identity: str = "license") -> pd.Series:
    """Build a provisional key independently from each normalized record."""
    if identity not in {"license", "license_address"}:
        raise ValueError("identity must be 'license' or 'license_address'.")

    def key(row):
        number = row.license_number
        if number:
            if identity == "license":
                return "license:" + number
            if not row.address:
                return "inspection:" + row.inspection_id
            location = [number, row.address.casefold(), row.city.casefold(),
                        row.state.casefold(), "" if row.zip == "UNKNOWN" else row.zip.casefold()]
            encoded = json.dumps(location, ensure_ascii=False).encode("utf-8")
            return "license_address:" + hashlib.sha256(encoded).hexdigest()[:24]
        if row.dba_name and row.address:
            # Matches preparation's fallback rule, including blank ZIP as blank.
            parts = [row.dba_name, row.address, row.city, row.state,
                     "" if row.zip == "UNKNOWN" else row.zip]
            encoded = "|".join(value.casefold() for value in parts).encode("utf-8")
            return "fallback:" + hashlib.sha256(encoded).hexdigest()[:24]
        return "inspection:" + row.inspection_id

    return pd.Series(
        [key(row) for row in data.itertuples(index=False)], index=data.index,
        dtype=object, name="establishment_key",
    )


def _history_features(target, source, cutoff):
    """Query cumulative daily history strictly before each lookup date."""
    columns = ["prior_visits", "prior_judgements", "prior_failures", "previous_outcome"]
    empty = pd.DataFrame(index=target.index)
    if source.empty or target.empty:
        for field in columns[:3]:
            empty[field] = np.zeros(len(target), dtype="int64")
        empty["previous_outcome"] = "No earlier judgement"
        return empty

    daily = source.groupby(["establishment_key", "inspection_date"], sort=True).agg(
        visits=("inspection_id", "size"), judgements=("label", "count"), failures=("label", "sum")
    )
    for before, after in zip(["visits", "judgements", "failures"], columns[:3]):
        daily[after] = daily[before].groupby(level=0).cumsum().astype("int64")
    status = pd.Series(np.nan, index=daily.index)
    status.loc[daily.judgements.gt(0) & daily.failures.eq(0)] = 0
    status.loc[daily.judgements.gt(0) & daily.failures.eq(daily.judgements)] = 1
    status.loc[daily.failures.gt(0) & daily.failures.lt(daily.judgements)] = 2
    daily["previous_outcome"] = status.groupby(level=0).ffill().map({
        0: "Previous non-Fail", 1: "Previous Fail", 2: "Mixed previous date",
    })
    left = target[["establishment_key"]].copy()
    left["lookup_date"] = cutoff
    left["_row"] = np.arange(len(left))
    right = daily[columns].reset_index()
    right["establishment_key"] = right.establishment_key.astype(object)
    matched = pd.merge_asof(
        left.sort_values("lookup_date"), right.sort_values("inspection_date"),
        left_on="lookup_date", right_on="inspection_date", by="establishment_key",
        direction="backward", allow_exact_matches=False,
    ).sort_values("_row").reset_index(drop=True)
    for field in columns[:3]:
        matched[field] = matched[field].fillna(0).astype("int64")
    matched["previous_outcome"] = matched.previous_outcome.fillna("No earlier judgement")
    return matched[columns]


def build_features(
    targets: pd.DataFrame, *, history: pd.DataFrame | None = None,
    batch_start=None, identity: str = "license", include_history: bool = True,
) -> FeatureSet:
    """Return aligned features, nullable labels and metadata.

    Training: history may contain all visits; every row queries only earlier dates.
    Batch evaluation: batch_start is the inclusive prediction-period start, and
    history is required. Use only visits strictly before that date.
    Returned indices are reset; meta.inspection_id preserves the input row order.
    """
    target = normalize_inspections(targets)
    if batch_start is not None and history is None:
        raise ValueError("An explicit history is required for batch prediction.")
    source_input = (targets if history is None else history).rename(
        columns=RAW_TO_CLEAN
    ).copy().reset_index(drop=True)
    if source_input.columns.duplicated().any():
        raise ValueError("Ambiguous raw/prepared history column names.")
    cutoff = target.inspection_date.copy()
    if batch_start is not None:
        start = pd.Timestamp(batch_start)
        if pd.isna(start) or start.tzinfo is not None or start != start.normalize():
            raise ValueError("batch_start must be a calendar date without a timezone.")
        if target.inspection_date.lt(start).any():
            raise ValueError("A target inspection precedes batch_start.")
        if "inspection_date" not in source_input:
            raise ValueError("Historical Inspection Date is required.")
        # Discard unavailable rows before reading labels, IDs or identities.
        source_input = source_input.loc[_dates(source_input.inspection_date).lt(start)]
        cutoff = pd.Series(start, index=target.index, dtype="datetime64[ns]")
    if not source_input.empty and not {"results", "label"}.intersection(source_input.columns):
        raise ValueError("Nonempty history must include Results or label, even when all are missing.")
    source = normalize_inspections(source_input)
    target["establishment_key"] = establishment_keys(target, identity)
    source["establishment_key"] = establishment_keys(source, identity)
    common = target[["inspection_id", "inspection_date", "establishment_key"]].merge(
        source[["inspection_id", "inspection_date", "establishment_key"]],
        on="inspection_id", suffixes=("_target", "_history"),
    )
    if (
        common.inspection_date_target.ne(common.inspection_date_history)
        | common.establishment_key_target.ne(common.establishment_key_history)
    ).any():
        raise ValueError("A shared Inspection ID has inconsistent date or identity.")

    past = _history_features(target, source, cutoff)
    X = target[["facility_type", "risk", "inspection_type", "zip"]].copy()
    X["inspection_month"] = target.inspection_date.dt.month.astype(str)
    X["inspection_weekday"] = target.inspection_date.dt.dayofweek.astype(str)
    X[BASE_NUMERIC_FEATURES] = target[BASE_NUMERIC_FEATURES]
    if include_history:
        X[HISTORY_NUMERIC_FEATURES[:3]] = past[HISTORY_NUMERIC_FEATURES[:3]]
        X["prior_fail_rate"] = past.prior_failures / past.prior_judgements.replace(0, np.nan)
        X["previous_outcome"] = past.previous_outcome
    meta = target[["inspection_id", "inspection_date", "establishment_key"]].copy()
    meta["history_before"] = cutoff
    meta["known_establishment"] = past.prior_visits.gt(0)
    return FeatureSet(X=X, y=target.label.copy(), meta=meta)


def build_prediction_features(
    targets: pd.DataFrame, *, history: pd.DataFrame, batch_start,
    identity: str = "license", include_history: bool = True,
) -> FeatureSet:
    """Build one unlabelled prediction batch from saved pre-batch history.

    Drop current outcome columns and return missing y values.
    Every target uses the same saved history from before batch_start.
    """
    if batch_start is None:
        raise ValueError("batch_start is required for prediction.")
    forbidden = {"results", "label", "violations", "violations_missing"}
    inputs = targets.drop(columns=[
        name for name in targets.columns if RAW_TO_CLEAN.get(name, name) in forbidden
    ])
    return build_features(
        inputs, history=history, batch_start=batch_start,
        identity=identity, include_history=include_history,
    )


def load_history(root: Path = ROOT) -> pd.DataFrame:
    """Read the existing history after verifying its frozen source and output hashes."""
    config = json.loads((root / "config/data.json").read_text(encoding="utf-8"))
    pin = json.loads((root / "config/assessment2_data.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / config["download_metadata"]).read_text(encoding="utf-8"))
    manifest = json.loads((root / "data/processed/preparation_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(root / config["raw_csv"]) != pin["raw_sha256"]:
        raise ValueError("Raw snapshot differs from the frozen Assessment 2 version.")
    if metadata["csv_sha256"] != pin["raw_sha256"] or metadata["downloaded_at_utc"] != pin["downloaded_at_utc"]:
        raise ValueError("Download metadata differs from the frozen version.")
    for group in ["inputs", "outputs"]:
        # Validate the metadata, preparation rules, and the history file being read.
        for relative, digest in manifest[group].items():
            if group == "inputs" or relative == config["processed_history"]:
                if sha256_file(root / relative) != digest:
                    raise ValueError(f"Preparation checksum mismatch: {relative}")
    if config["processed_history"] not in manifest["outputs"]:
        raise ValueError("Prepared history is missing from the checksum manifest.")
    fields = [
        "inspection_id", "inspection_date", "dba_name", "license_number",
        "address", "city", "state", "zip", "facility_type", "risk",
        "inspection_type", "latitude", "longitude", "label",
    ]
    frame = pd.read_csv(root / config["processed_history"], usecols=fields,
                        dtype=str, keep_default_na=False)
    result = normalize_inspections(frame)
    actual = {"start": str(result.inspection_date.min().date()),
              "end": str(result.inspection_date.max().date())}
    if len(result) != pin["row_count"] or actual != pin["actual_date_range"]:
        raise ValueError("Prepared history does not match the pinned count/date range.")
    return result


if __name__ == "__main__":
    history = load_history()
    validation_features = {}
    for identity in ["license", "license_address"]:
        train = history.loc[history.inspection_date.le("2024-12-31")]
        validation = history.loc[history.inspection_date.between("2025-01-01", "2025-12-31")]
        training = build_features(train, history=train, identity=identity)
        batch = build_features(validation, history=train, batch_start="2025-01-01", identity=identity)
        validation_features[identity] = batch
        print(f"{identity}: training rows={len(training.X):,}, labelled={training.y.notna().sum():,}; "
              f"validation rows={len(batch.X):,}, labelled={batch.y.notna().sum():,}; "
              f"features={len(batch.X.columns)}")
        print(f"  Known validation rows: {batch.meta.loc[batch.y.notna(), 'known_establishment'].sum():,}; "
              f"missing history rates: {batch.X.prior_fail_rate.isna().sum():,}.")
    primary, alternate = validation_features["license"], validation_features["license_address"]
    fields = HISTORY_NUMERIC_FEATURES + HISTORY_CATEGORICAL_FEATURES
    same = primary.X[fields].eq(alternate.X[fields]) | (
        primary.X[fields].isna() & alternate.X[fields].isna()
    )
    changed = ~same.all(axis=1) & primary.y.notna()
    print(f"Identity choice changes history for {changed.sum():,} labelled validation rows.")
    print("Feature checks complete for training and 2025 validation data.")
