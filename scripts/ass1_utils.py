"""Small data and descriptive-statistics helpers for Assessment 1."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_data import RAW_TO_CLEAN, normalize_row


def load_snapshot(root: Path):
    config = json.loads((root / "config/data.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / config["download_metadata"]).read_text(encoding="utf-8"))
    path = root / config["raw_csv"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != metadata["csv_sha256"]:
        raise ValueError("The saved raw snapshot does not match its recorded SHA-256.")
    if pd.Timestamp(metadata["downloaded_at_utc"]).date() > pd.Timestamp(config["history_cutoff"]).date():
        raise ValueError("The snapshot was downloaded after the project cutoff.")
    raw = pd.read_csv(path, dtype="string", keep_default_na=False, na_values=[""])
    if list(raw.columns) != list(RAW_TO_CLEAN) or len(raw) != metadata["row_count"]:
        raise ValueError("Unexpected raw schema or row count.")
    dates = pd.to_datetime(raw["Inspection Date"], format="%m/%d/%Y", errors="raise")
    if not dates.between(config["history_start"], config["history_cutoff"]).all():
        raise ValueError("An inspection lies outside the permitted historical period.")
    if not dates.between(metadata["requested_window"]["start_date"], metadata["requested_window"]["end_date"]).all():
        raise ValueError("An inspection lies outside the saved download window.")
    return raw, dates, metadata


def clean_history(raw: pd.DataFrame) -> tuple[pd.DataFrame, Counter]:
    """Reuse preparation rules, retaining evidence rather than fitting a transform."""
    unique = raw.drop_duplicates().copy()
    if unique["Inspection ID"].duplicated().any():
        raise ValueError("Conflicting duplicate IDs need an explicit resolution.")
    counts = Counter()
    fields = [name for name in raw.columns if name not in {"Violations", "Location"}]

    def records():
        for values in unique[fields].itertuples(index=False, name=None):
            record = dict(zip(fields, (value if isinstance(value, str) else "" for value in values)))
            normalized = normalize_row(record, counts)
            yield {name: value for name, value in normalized.items() if name not in {"violations", "location"}}

    history = pd.DataFrame.from_records(records())
    history["inspection_date"] = pd.to_datetime(history["inspection_date"], format="%Y-%m-%d")
    for field in ["label", "risk_level"]:
        history[field] = pd.to_numeric(history[field], errors="coerce").astype("Int8")
    for field in ["latitude", "longitude"]:
        history[field] = pd.to_numeric(history[field], errors="coerce")
    history["violations_missing"] = unique["Violations"].str.fullmatch(r"\s*", na=True).to_numpy(dtype=bool)
    return history, counts


def group_rates(data: pd.DataFrame, field: str) -> pd.DataFrame:
    """Inspection-level Wilson intervals are descriptive with repeated venues."""
    labelled = data.loc[data["label"].notna()]
    rates = labelled.groupby(field, observed=True, dropna=False)["label"].agg(n="size", failures="sum")
    rates["failures"] = rates["failures"].astype(int)
    n, proportion, z = rates["n"].astype(float), rates["failures"] / rates["n"], 1.959963984540054
    denominator = 1 + z*z/n
    center = (proportion + z*z/(2*n)) / denominator
    half = z * np.sqrt(proportion*(1-proportion)/n + z*z/(4*n*n)) / denominator
    rates["rate"] = proportion.astype(float)
    rates["low"], rates["high"] = (center-half).astype(float), (center+half).astype(float)
    return rates


def strict_past_features(history: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each day before shifting, so same-day outcomes cannot leak."""
    keys = ["establishment_key", "inspection_date"]
    daily = history.groupby(keys, observed=True, sort=True).agg(
        visits=("inspection_id", "size"), judgements=("label", "count"), failures=("label", "sum")
    )
    grouped = daily.groupby(level=0, sort=False)
    daily["prior_visits"] = grouped["visits"].cumsum() - daily["visits"]
    daily["prior_judgements"] = grouped["judgements"].cumsum() - daily["judgements"]
    daily["prior_failures"] = grouped["failures"].cumsum() - daily["failures"]
    code = pd.Series(np.nan, index=daily.index)
    code.loc[(daily.judgements > 0) & (daily.failures == 0)] = 0
    code.loc[(daily.judgements > 0) & (daily.failures == daily.judgements)] = 1
    code.loc[(daily.failures > 0) & (daily.failures < daily.judgements)] = 2
    previous_code = code.groupby(level=0).ffill().groupby(level=0).shift(1)
    daily["previous_outcome"] = previous_code.map({
        0: "Previous non-Fail", 1: "Previous Fail", 2: "Mixed previous date"
    }).fillna("No earlier judgement")
    columns = ["prior_visits", "prior_judgements", "prior_failures", "previous_outcome"]
    return history[["inspection_id", *keys]].merge(
        daily[columns].reset_index(), on=keys, how="left", validate="many_to_one"
    )[["inspection_id", *columns]]
