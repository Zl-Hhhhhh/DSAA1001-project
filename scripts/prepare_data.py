#!/usr/bin/env python3
"""Prepare saved inspection data offline; never fetch data or train a model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import date, datetime
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config/data.json").read_text(encoding="utf-8"))
RAW_TO_CLEAN = {
    "Inspection ID": "inspection_id", "DBA Name": "dba_name", "AKA Name": "aka_name",
    "License #": "license_number", "Facility Type": "facility_type", "Risk": "risk",
    "Address": "address", "City": "city", "State": "state", "Zip": "zip",
    "Inspection Date": "inspection_date", "Inspection Type": "inspection_type",
    "Results": "results", "Violations": "violations", "Latitude": "latitude",
    "Longitude": "longitude", "Location": "location",
}
HISTORY_COLUMNS = list(RAW_TO_CLEAN.values()) + [
    "risk_level", "establishment_key", "inspection_year", "inspection_month",
    "inspection_weekday", "label",
]
LABELLED_COLUMNS = [
    name for name in HISTORY_COLUMNS if name not in {"results", "violations", "location"}
]
DEFAULT_FEATURES = [
    "facility_type", "risk", "risk_level", "inspection_type", "city", "state", "zip",
    "latitude", "longitude", "inspection_year", "inspection_month", "inspection_weekday",
]
CATEGORY_FIELDS = ["Facility Type", "Risk", "Inspection Type", "Results", "City", "State"]
KNOWN_RESULTS = {value.casefold(): value for value in CONFIG["label_map"]}
KNOWN_RESULTS.update({value.casefold(): value for value in CONFIG["excluded_results"]})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tidy(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def identifier(value: str, *, positive: bool = True) -> str:
    value = re.sub(r"\.0$", "", tidy(value))
    if not value.isdigit() or (positive and int(value) <= 0):
        return ""
    return value


def normalize_row(raw: dict, counts: Counter) -> dict:
    row = {clean: (raw.get(original) or "") for original, clean in RAW_TO_CLEAN.items()}
    row["inspection_id"] = identifier(row["inspection_id"])
    if not row["inspection_id"]:
        raise ValueError("Missing or invalid Inspection ID; cannot safely match records.")
    inspected_on = datetime.strptime(tidy(row["inspection_date"]), "%m/%d/%Y").date()
    if not date.fromisoformat(CONFIG["history_start"]) <= inspected_on <= date.fromisoformat(CONFIG["history_cutoff"]):
        raise ValueError(f"Inspection {row['inspection_id']} is outside the permitted history window.")
    row["inspection_date"] = inspected_on.isoformat()
    for column in ["dba_name", "aka_name", "address", "zip", "location"]:
        row[column] = tidy(row[column])
    for column in ["facility_type", "risk", "inspection_type", "city", "state"]:
        row[column] = tidy(row[column]).upper()
    for column in ["facility_type", "risk", "inspection_type"]:
        if not row[column]:
            row[column] = "UNKNOWN"
    raw_license = tidy(row["license_number"])
    row["license_number"] = identifier(raw_license)
    if not row["license_number"]:
        counts["missing_or_invalid_license"] += 1
        if raw_license == "0":
            counts["zero_license"] += 1
    row["zip"] = re.sub(r"^(\d{5})\.0$", r"\1", row["zip"])
    if row["zip"] and not re.fullmatch(r"\d{5}(?:-\d{4})?", row["zip"]):
        counts["nonstandard_zip"] += 1
    for column, limit in [("latitude", 90), ("longitude", 180)]:
        original = tidy(row[column])
        if original:
            try:
                number = float(original)
                if not math.isfinite(number) or not -limit <= number <= limit:
                    raise ValueError
                row[column] = str(number)
            except ValueError:
                counts[f"invalid_{column}"] += 1
                row[column] = ""
        else:
            row[column] = ""
    result = tidy(row["results"])
    if result:
        canonical = KNOWN_RESULTS.get(result.casefold())
        if canonical is None:
            raise ValueError(f"Unrecognised Results value {result!r}; review its meaning before assigning labels.")
        row["results"] = canonical
    else:
        row["results"] = ""
        counts["missing_result"] += 1
    row["label"] = CONFIG["label_map"].get(row["results"], "")
    if row["license_number"]:
        row["establishment_key"] = "license:" + row["license_number"]
    elif row["dba_name"] and row["address"]:
        identity = "|".join(row[name].casefold() for name in ["dba_name", "address", "city", "state", "zip"])
        row["establishment_key"] = "fallback:" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        counts["fallback_identity"] += 1
    else:
        row["establishment_key"] = "inspection:" + row["inspection_id"]
        counts["inspection_only_identity"] += 1
    match = re.fullmatch(r"RISK ([123]) \(.+\)", row["risk"])
    row["risk_level"] = int(match.group(1)) if match else ""
    row["inspection_year"] = inspected_on.year
    row["inspection_month"] = inspected_on.month
    row["inspection_weekday"] = inspected_on.weekday()
    return row


def compressed_writer(path: Path, columns: list[str]):
    binary = path.open("wb")
    compressed = gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    writer = csv.DictWriter(text, fieldnames=columns)
    writer.writeheader()
    return binary, text, writer


def write_csv(path: Path, columns: list[str], rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def snapshot_markdown(report: dict) -> str:
    metadata = report["source"]
    lines = [
        f"- 数据来源：[Chicago Food Inspections]({metadata['portal_page']})（{metadata['dataset_id']}）。",
        f"- 下载时间（UTC）：**{metadata['downloaded_at_utc']}**。",
        f"- 本次请求范围：{metadata['requested_window']['start_date']} 至 {metadata['requested_window']['end_date']}。",
        f"- 实际记录日期：**{report['date_range']['start']} 至 {report['date_range']['end']}**。",
        f"- 原始记录：**{report['raw_rows']:,}**；去除完全重复后：**{report['history_rows']:,}**。",
        f"- 有标签记录：**{report['labelled_rows']:,}**；无 Pass/Fail 判断的历史记录：**{report['unlabelled_rows']:,}**。",
        f"- 标签 0：**{report['class_counts']['0']:,}**；标签 1：**{report['class_counts']['1']:,}**"
        f"（Fail 占有标签记录 **{report['fail_rate_percent']:.2f}%**）。",
        f"- 重复 Inspection ID：{report['duplicate_id_rows']:,}；完全重复并移除：{report['exact_duplicate_rows_removed']:,}。",
        f"- 多次检查的初步店铺分组：{report['repeated_establishment_groups']:,}。",
        "- 请求截止日期不代表数据已完整上报；此快照不包含尚未公开的记录，不能称为截至 10 月 30 日的完整数据。",
        "", "原始字段缺失率（空值或仅空白字符）：", "",
        "| 原始字段 | 缺失数 | 缺失率 |", "|---|---:|---:|",
    ]
    for name, stats in report["raw_missing"].items():
        lines.append(f"| {name} | {stats['count']:,} | {stats['percent']:.2f}% |")
    return "\n".join(lines) + "\n"


def prepare(raw_path: Path, metadata_path: Path, output_root: Path) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset_id") != CONFIG["dataset_id"]:
        raise ValueError("Snapshot metadata identifies the wrong dataset.")
    downloaded_at = datetime.fromisoformat(metadata["downloaded_at_utc"])
    if downloaded_at.tzinfo is None or downloaded_at.date() > date.fromisoformat(CONFIG["history_cutoff"]):
        raise ValueError("Snapshot was not recorded with a valid pre-cutoff download timestamp.")
    if sha256_file(raw_path) != metadata["csv_sha256"]:
        raise ValueError("Raw data checksum mismatch; restore the saved snapshot before preparing data.")
    requested_start = date.fromisoformat(metadata["requested_window"]["start_date"])
    requested_end = date.fromisoformat(metadata["requested_window"]["end_date"])
    if not date.fromisoformat(CONFIG["history_start"]) <= requested_start <= requested_end <= min(
        date.fromisoformat(CONFIG["history_cutoff"]), downloaded_at.date()
    ):
        raise ValueError("Metadata requests a forbidden history period.")
    processed, reports = output_root / "data/processed", output_root / "reports"
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    history_path, labelled_path = output_root / CONFIG["processed_history"], output_root / CONFIG["labelled_data"]
    history_partial = history_path.with_name(history_path.name + ".partial")
    labelled_partial = labelled_path.with_name(labelled_path.name + ".partial")
    counters, missing_raw, missing_clean, class_counts = Counter(), Counter(), Counter(), Counter()
    raw_categories = {name: Counter() for name in CATEGORY_FIELDS}
    clean_categories = {RAW_TO_CLEAN[name]: Counter() for name in CATEGORY_FIELDS}
    seen_ids, establishments, license_addresses = {}, Counter(), defaultdict(set)
    same_day_counts, same_day_labels = Counter(), defaultdict(set)
    yearly_counts, result_counts = defaultdict(Counter), Counter()
    labelled_rows = []
    min_date, max_date = None, None
    binary, history_text, history_writer = compressed_writer(history_partial, HISTORY_COLUMNS)
    try:
        with gzip.open(raw_path, "rt", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames != list(RAW_TO_CLEAN):
                raise ValueError("Raw CSV schema differs from the expected 17 portal fields.")
            for raw in reader:
                if None in raw:
                    raise ValueError("Malformed CSV record has extra fields.")
                counters["raw_rows"] += 1
                for column in RAW_TO_CLEAN:
                    if not tidy(raw.get(column)):
                        missing_raw[column] += 1
                for column in CATEGORY_FIELDS:
                    raw_categories[column][raw.get(column) or ""] += 1
                inspection_id = identifier(raw["Inspection ID"])
                signature = hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
                if inspection_id in seen_ids:
                    counters["duplicate_id_rows"] += 1
                    if seen_ids[inspection_id] != signature:
                        raise ValueError(f"Conflicting records share Inspection ID {inspection_id}; resolve explicitly.")
                    counters["exact_duplicate_rows_removed"] += 1
                    continue
                seen_ids[inspection_id] = signature
                row = normalize_row(raw, counters)
                inspected_on = date.fromisoformat(row["inspection_date"])
                if not requested_start <= inspected_on <= requested_end:
                    raise ValueError("A record is outside the metadata's requested window.")
                min_date = inspected_on if min_date is None else min(min_date, inspected_on)
                max_date = inspected_on if max_date is None else max(max_date, inspected_on)
                history_writer.writerow(row)
                counters["history_rows"] += 1
                for column in HISTORY_COLUMNS:
                    if row[column] == "":
                        missing_clean[column] += 1
                for column in clean_categories:
                    clean_categories[column][row[column]] += 1
                result_counts[row["results"]] += 1
                establishments[row["establishment_key"]] += 1
                if row["license_number"]:
                    license_addresses[row["license_number"]].add(row["address"].casefold())
                same_day = (row["establishment_key"], row["inspection_date"])
                same_day_counts[same_day] += 1
                yearly_counts[inspected_on.year]["history"] += 1
                if row["label"] != "":
                    class_counts[row["label"]] += 1
                    same_day_labels[same_day].add(row["label"])
                    yearly_counts[inspected_on.year]["labelled"] += 1
                    yearly_counts[inspected_on.year]["fail"] += row["label"]
                    labelled_rows.append({name: row[name] for name in LABELLED_COLUMNS})
                if counters["raw_rows"] % 50000 == 0:
                    print(f"Prepared {counters['raw_rows']:,} raw rows", flush=True)
        if counters["raw_rows"] != metadata["row_count"]:
            raise ValueError("Raw CSV row count disagrees with download metadata.")
        if not counters["history_rows"] or not labelled_rows:
            raise ValueError("Snapshot has no usable history or no labelled inspections.")
    except BaseException:
        history_text.close()
        binary.close()
        history_partial.unlink(missing_ok=True)
        raise
    history_text.close()
    binary.close()
    labelled_rows.sort(key=lambda row: (row["inspection_date"], int(row["inspection_id"])))
    labelled_binary, labelled_text, labelled_writer = compressed_writer(labelled_partial, LABELLED_COLUMNS)
    try:
        labelled_writer.writerows(labelled_rows)
    finally:
        labelled_text.close()
        labelled_binary.close()
    history_partial.replace(history_path)
    labelled_partial.replace(labelled_path)
    raw_count, history_count = counters["raw_rows"], counters["history_rows"]
    report = {
        "preparation_version": 1, "source": metadata,
        "raw_rows": raw_count, "history_rows": history_count,
        "labelled_rows": len(labelled_rows), "unlabelled_rows": history_count - len(labelled_rows),
        "class_counts": {"0": class_counts[0], "1": class_counts[1]},
        "fail_rate_percent": 100 * class_counts[1] / len(labelled_rows),
        "date_range": {"start": min_date.isoformat(), "end": max_date.isoformat()},
        "duplicate_id_rows": counters["duplicate_id_rows"],
        "exact_duplicate_rows_removed": counters["exact_duplicate_rows_removed"],
        "establishment_groups": len(establishments),
        "repeated_establishment_groups": sum(count > 1 for count in establishments.values()),
        "licenses_with_multiple_addresses": sum(len(addresses) > 1 for addresses in license_addresses.values()),
        "same_day_multiple_inspection_groups": sum(count > 1 for count in same_day_counts.values()),
        "same_day_conflicting_label_groups": sum(len(labels) > 1 for labels in same_day_labels.values()),
        "normalization_counts": dict(counters), "result_counts": dict(result_counts),
        "raw_missing": {
            name: {"count": missing_raw[name], "percent": 100 * missing_raw[name] / raw_count}
            for name in RAW_TO_CLEAN
        },
        "processed_missing": {
            name: {"count": missing_clean[name], "percent": 100 * missing_clean[name] / history_count}
            for name in HISTORY_COLUMNS
        },
        "category_cardinality": {
            original: {"raw": len(raw_categories[original]), "normalized": len(clean_categories[RAW_TO_CLEAN[original]])}
            for original in CATEGORY_FIELDS
        },
        "default_feature_columns": DEFAULT_FEATURES, "labelled_columns": LABELLED_COLUMNS,
        "identity_rule": "Valid licence first; otherwise hash of name/address/city/state/ZIP; otherwise inspection-specific key.",
        "history_rule": "Only strictly earlier dates may contribute outcomes to later-inspection features; same-day ordering is unknown.",
        "ordering": {"history": "saved raw input order", "labelled": "inspection_date ascending, numeric inspection_id ascending"},
    }
    (reports / "data_quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(reports / "missing_values.csv", ["table", "column", "missing_count", "missing_percent"], [
        {"table": table, "column": column, "missing_count": stats["count"], "missing_percent": round(stats["percent"], 6)}
        for table, values in [("raw", report["raw_missing"]), ("processed_history", report["processed_missing"])]
        for column, stats in values.items()
    ])
    write_csv(reports / "category_counts.csv", ["table", "column", "value", "count"], [
        {"table": table, "column": column, "value": value, "count": count}
        for table, fields in [("raw", raw_categories), ("processed_history", clean_categories)]
        for column, values in fields.items() for value, count in sorted(values.items())
    ])
    write_csv(reports / "annual_counts.csv", ["year", "history_rows", "labelled_rows", "fail_rows"], [
        {"year": year, "history_rows": values["history"], "labelled_rows": values["labelled"], "fail_rows": values["fail"]}
        for year, values in sorted(yearly_counts.items())
    ])
    write_csv(reports / "label_counts.csv", ["label", "count", "percent"], [
        {"label": label, "count": class_counts[label], "percent": round(100 * class_counts[label] / len(labelled_rows), 6)}
        for label in [0, 1]
    ])
    availability = []
    for column in HISTORY_COLUMNS:
        role = "candidate_feature" if column in DEFAULT_FEATURES else "metadata"
        rule = "Use only if supplied/known before inspection; fit statistical preprocessing on training data only."
        available = "yes"
        if column == "label":
            role, available, rule = "target", "no", "Fail=1; Pass and Pass w/ Conditions=0; other outcomes have no label."
        elif column in {"results", "violations"}:
            role, available, rule = "history_only_outcome", "no", "Current value is forbidden as input; strictly earlier inspection values may be used as history."
        elif column == "establishment_key":
            role, rule = "grouping_key", "Preliminary identity; licence reuse, changed addresses and fallback matching need review."
        elif column in {"inspection_id", "license_number"}:
            rule = "Identifier, not a continuous numeric feature."
        availability.append({"column": column, "role": role, "available_before_current_inspection": available, "rule": rule})
    write_csv(reports / "feature_availability.csv", list(availability[0]), availability)
    summary = "# 数据准备报告\n\n" + snapshot_markdown(report)
    summary += (
        "\n## 处理规则\n\n"
        "- 原始 CSV 以 gzip 保存；保留全部 17 个字段，包括本次检查结果和违规描述。\n"
        "- 历史表保留无 Pass/Fail 判断的记录。带标签表仅保留 Fail / Pass / Pass w/ Conditions，不包含 results、violations 或 location。\n"
        "- 日期明确按 MM/DD/YYYY 解析并输出 YYYY-MM-DD；编号和邮编作为字符串处理。\n"
        "- 类别只清理首尾/连续空白并统一大小写，不做未经证据支持的拼写合并。\n"
        "- 缺失 facility_type / risk / inspection_type 使用 UNKNOWN；不推测缺失州名、城市或邮编，不做全数据统计填补。\n"
        "- 仅移除同一 Inspection ID 的完全重复记录；相同 ID 内容冲突则停止。重复店铺访问及同日检查均保留。\n"
        "- 无效/为 0 的许可证不作为共同店铺键；回退键需要后续 EDA 核查。同一许可证跨地址不能自动视为可靠的唯一店铺。\n"
        "- 坐标仅校验全球合法数值范围；未筛掉郊区记录，未判断坐标是否确实对应地址。\n"
        "- 仅生成静态/日历候选变量；尚未构造历史结果特征、划分训练集或训练模型。\n"
        "- 历史特征必须使用严格早于当前检查日期的记录；同日先后未知，不得借 Inspection ID 推断发生顺序。\n"
        "- 官方元数据提示 2018-07-01 检查程序变更；后续 EDA 应考察其影响。当前快照还可能有上报延迟及回溯修订。\n"
    )
    summary += (
        f"\n## 需要后续核查的情况\n\n"
        f"- 无效/缺失许可证：{counters['missing_or_invalid_license']:,} 条，其中为 0：{counters['zero_license']:,} 条。\n"
        f"- 名称和地址回退分组：{counters['fallback_identity']:,} 条；仅按 Inspection ID 分组：{counters['inspection_only_identity']:,} 条。\n"
        f"- 同一许可证对应多个地址：{report['licenses_with_multiple_addresses']:,} 组。\n"
        f"- 同日多次检查：{report['same_day_multiple_inspection_groups']:,} 组；同日同时出现标签 0 与 1：{report['same_day_conflicting_label_groups']:,} 组。\n"
        f"- 非标准 ZIP：{counters['nonstandard_zip']:,} 条；非法纬度：{counters['invalid_latitude']:,} 条；非法经度：{counters['invalid_longitude']:,} 条。\n"
    )
    (reports / "data_preparation.md").write_text(summary, encoding="utf-8")
    readme_path = output_root / "README.md"
    if readme_path.exists():
        readme = readme_path.read_text(encoding="utf-8")
        start, end = "<!-- DATA-SNAPSHOT:START -->", "<!-- DATA-SNAPSHOT:END -->"
        if start in readme and end in readme:
            before, tail = readme.split(start, 1)
            _, after = tail.split(end, 1)
            readme_path.write_text(before + start + "\n\n" + snapshot_markdown(report) + "\n" + end + after, encoding="utf-8")
    artifact_paths = [history_path, labelled_path] + [
        reports / name for name in [
            "data_quality.json", "missing_values.csv", "category_counts.csv", "annual_counts.csv",
            "label_counts.csv", "feature_availability.csv", "data_preparation.md",
        ]
    ]
    source_metadata_path = output_root / CONFIG["source_metadata"]
    input_paths = [raw_path, metadata_path] + ([source_metadata_path] if source_metadata_path.exists() else [])
    input_paths += [path for path in [output_root / "config/data.json", output_root / "scripts/prepare_data.py"] if path.exists()]
    manifest = {
        "version": 1,
        "inputs": {str(path.relative_to(output_root)).replace("\\", "/"): sha256_file(path) for path in input_paths},
        "outputs": {str(path.relative_to(output_root)).replace("\\", "/"): sha256_file(path) for path in artifact_paths},
        "history_rows": history_count, "labelled_rows": len(labelled_rows),
    }
    (processed / "preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {history_count:,} history rows; {len(labelled_rows):,} labelled rows; {class_counts[1]:,} Fail", flush=True)
    return report


def check(output_root: Path) -> None:
    manifest = json.loads((output_root / "data/processed/preparation_manifest.json").read_text(encoding="utf-8"))
    for group in ["inputs", "outputs"]:
        for relative, expected in manifest[group].items():
            path = output_root / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Checksum validation failed: {relative}")
    with gzip.open(output_root / CONFIG["labelled_data"], "rt", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != LABELLED_COLUMNS or {"results", "violations"} & set(reader.fieldnames):
            raise ValueError("Labelled data has an unexpected or leaking schema.")
        count, previous = 0, None
        for row in reader:
            if row["label"] not in {"0", "1"}:
                raise ValueError("Invalid binary label")
            current = (row["inspection_date"], int(row["inspection_id"]))
            if previous is not None and current < previous:
                raise ValueError("Labelled data is not ordered chronologically.")
            previous = current
            count += 1
        if count != manifest["labelled_rows"]:
            raise ValueError("Labelled row count does not match manifest")
    print(f"Verified all input/output SHA-256 checksums and {count:,} chronologically ordered binary labels.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify without rewriting files")
    args = parser.parse_args()
    if args.check:
        check(ROOT)
    else:
        prepare(ROOT / CONFIG["raw_csv"], ROOT / CONFIG["download_metadata"], ROOT)
        check(ROOT)


if __name__ == "__main__":
    main()
