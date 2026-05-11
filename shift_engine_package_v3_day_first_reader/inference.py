
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import lightgbm as lgb


ENGINE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ENGINE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "lightgbm_model.txt"
FEATURE_COLS_PATH = ARTIFACT_DIR / "feature_cols.json"
LABEL_CLASSES_PATH = ARTIFACT_DIR / "label_classes.json"
SHIFT_META_PATH = ARTIFACT_DIR / "shift_meta.json"
CONFIG_PATH = ARTIFACT_DIR / "config.json"


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


FEATURE_COLS = _load_json(FEATURE_COLS_PATH)
LABEL_CLASSES = _load_json(LABEL_CLASSES_PATH)
SHIFT_META = _load_json(SHIFT_META_PATH)
CONFIG = _load_json(CONFIG_PATH)

MODEL = lgb.Booster(model_file=str(MODEL_PATH))

MODELED_SHIFT_CLASSES = list(LABEL_CLASSES)

WINDOW_HOURS_AFTER = int(CONFIG.get("window_hours_after", 48))
MIN_VALID_SPAN = int(CONFIG.get("min_valid_span", 120))
MAX_NORMAL_SPAN = int(CONFIG.get("max_normal_span", 16 * 60))
MAX_EXTENDED_SPAN = int(CONFIG.get("max_extended_span", 24 * 60))
MAX_LONG_DUTY_SPAN = int(CONFIG.get("max_long_duty_span", 48 * 60))

TIME_BUCKETS = [
    (0, 300, "cnt_00_05"),
    (300, 480, "cnt_05_08"),
    (480, 720, "cnt_08_12"),
    (720, 840, "cnt_12_14"),
    (840, 1080, "cnt_14_18"),
    (1080, 1260, "cnt_18_21"),
    (1260, 1440, "cnt_21_24"),
    (1440, 1740, "cnt_next_00_05"),
    (1740, 1920, "cnt_next_05_08"),
    (1920, 2160, "cnt_next_08_12"),
]


def _read_input_file(file_path):
    file_path = str(file_path)
    lower = file_path.lower()

    if lower.endswith(".csv"):
        return pd.read_csv(file_path, low_memory=False)

    if lower.endswith(".xls") or lower.endswith(".xlsx"):
        try:
            df = pd.read_csv(file_path, sep="\t", low_memory=False)
            if df.shape[1] > 1:
                return df
        except Exception:
            pass

        return pd.read_excel(file_path)

    try:
        return pd.read_csv(file_path, sep="\t", low_memory=False)
    except Exception:
        return pd.read_csv(file_path, low_memory=False)


def _normalize_columns(df):
    cols = {c.lower().strip(): c for c in df.columns}

    emp_col = None
    for c in ["empcode", "personcode", "employeeid", "employee_id", "personcode_norm", "empcode_norm"]:
        if c in cols:
            emp_col = cols[c]
            break

    dt_col = None
    for c in ["transactiondatetime", "transaction_date_time", "datetime", "punchdatetime", "punch_time", "punchtime", "txndatetime"]:
        if c in cols:
            dt_col = cols[c]
            break

    reader_col = None
    for c in ["readernumber", "reader_number"]:
        if c in cols:
            reader_col = cols[c]
            break

    if emp_col is None:
        raise ValueError("Missing required employee column. Expected EmpCode or PersonCode.")

    if dt_col is None:
        raise ValueError("Missing required datetime column. Expected TransactionDateTime.")

    if reader_col is None:
        raise ValueError("Missing required column: ReaderNumber. V3 day-first reader engine requires ReaderNumber where 1 = IN and 2 = OUT.")

    out = pd.DataFrame()
    out["EmpCode_norm"] = df[emp_col].astype(str).str.strip()
    out["TransactionDateTime"] = pd.to_datetime(df[dt_col], errors="coerce")
    out["ReaderNumber"] = pd.to_numeric(df[reader_col], errors="coerce")

    for optional_col in ["ReaderId", "ReasonCode", "TransactionCode", "PersonName"]:
        if optional_col in df.columns:
            out[optional_col] = df[optional_col]
        else:
            out[optional_col] = np.nan

    out = out[out["TransactionDateTime"].notna()].copy()
    out = out[out["EmpCode_norm"].astype(str).str.strip() != ""].copy()

    out["txn_day"] = out["TransactionDateTime"].dt.normalize()
    out = out.sort_values(["EmpCode_norm", "TransactionDateTime"]).reset_index(drop=True)

    return out


def _build_calendar(txn):
    if txn.empty:
        return pd.DataFrame(columns=["PersonCode_norm", "attendance_day", "PersonCode"])

    employees = sorted(txn["EmpCode_norm"].dropna().astype(str).unique().tolist())
    min_day = txn["txn_day"].min()
    max_day = txn["txn_day"].max()
    dates = pd.date_range(min_day, max_day, freq="D")

    calendar = pd.MultiIndex.from_product(
        [employees, dates],
        names=["PersonCode_norm", "attendance_day"]
    ).to_frame(index=False)

    calendar["PersonCode"] = calendar["PersonCode_norm"]
    return calendar


def _build_daily_summary(txn):
    if txn.empty:
        return pd.DataFrame(columns=["PersonCode_norm", "attendance_day", "txn_punch_count"])

    txn["reader_is_in"] = (txn["ReaderNumber"] == 1).astype(int)
    txn["reader_is_out"] = (txn["ReaderNumber"] == 2).astype(int)

    daily = (
        txn.groupby(["EmpCode_norm", "txn_day"], as_index=False)
        .agg(
            txn_punch_count=("TransactionDateTime", "size"),
            txn_first_punch=("TransactionDateTime", "min"),
            txn_last_punch=("TransactionDateTime", "max"),
            txn_in_reader_count=("reader_is_in", "sum"),
            txn_out_reader_count=("reader_is_out", "sum"),
        )
        .rename(columns={
            "EmpCode_norm": "PersonCode_norm",
            "txn_day": "attendance_day",
        })
    )

    return daily


def _build_txn_index(txn):
    index = {}

    for emp, g in txn.groupby("EmpCode_norm", sort=False):
        g = g.sort_values("TransactionDateTime").reset_index(drop=True)

        index[emp] = {
            "ts": g["TransactionDateTime"].to_numpy(dtype="datetime64[ns]"),
            "reader_number": g["ReaderNumber"].to_numpy(),
            "reader": g["ReaderId"].to_numpy(),
            "reason": g["ReasonCode"].to_numpy(),
            "txn_code": g["TransactionCode"].to_numpy(),
        }

    return index


def _minutes_from_day0(ts, day0):
    return float((pd.Timestamp(ts) - pd.Timestamp(day0)).total_seconds() / 60.0)


def _get_window_df(txn_index, emp, day0):
    if pd.isna(day0) or emp not in txn_index:
        return pd.DataFrame(columns=["ts", "reader_number", "reader", "reason", "txn_code"])

    arr = txn_index[emp]["ts"]
    reader_numbers = txn_index[emp]["reader_number"]
    readers = txn_index[emp]["reader"]
    reasons = txn_index[emp]["reason"]
    txcodes = txn_index[emp]["txn_code"]

    start = np.datetime64(pd.Timestamp(day0))
    end = np.datetime64(pd.Timestamp(day0) + pd.Timedelta(hours=WINDOW_HOURS_AFTER))

    left = np.searchsorted(arr, start, side="left")
    right = np.searchsorted(arr, end, side="left")

    if right <= left:
        return pd.DataFrame(columns=["ts", "reader_number", "reader", "reason", "txn_code"])

    win = pd.DataFrame({
        "ts": pd.to_datetime(arr[left:right]),
        "reader_number": reader_numbers[left:right],
        "reader": readers[left:right],
        "reason": reasons[left:right],
        "txn_code": txcodes[left:right],
    }).sort_values("ts").reset_index(drop=True)

    win["pos"] = np.arange(len(win))
    win["minute_from_day0"] = win["ts"].apply(lambda x: _minutes_from_day0(x, day0)).astype(float)

    return win


def _build_gap_stats(ts_series):
    ts_series = pd.to_datetime(ts_series).sort_values().reset_index(drop=True)

    if len(ts_series) < 2:
        return 0.0, 0.0, 0

    gaps = np.diff(
        ts_series.values.astype("datetime64[s]")
    ).astype("timedelta64[m]").astype(int)

    return float(gaps.max()), float(gaps.mean()), int((gaps > 240).sum())


def _bucket_counts(mins):
    out = {}
    mins = pd.Series(mins).astype(float)

    for start_min, end_min, name in TIME_BUCKETS:
        out[name] = int(((mins >= start_min) & (mins < end_min)).sum())

    return out


def _format_ts_value(value):
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _join_limited_timestamps(values, limit=12):
    if values is None or len(values) == 0:
        return ""
    return " | ".join(_format_ts_value(value) for value in list(values)[:limit])


def _empty_fast_feature_frame(index):
    df = pd.DataFrame(index=index)
    defaults = {
        "window_punch_count": 0,
        "window_large_gap_count": 0,
        "window_cluster_count": 0,
        "reader_in_count": 0,
        "reader_out_count": 0,
        "reader_pair_found_flag": 0,
        "best_shift_candidate": np.nan,
        "best_shift_score": np.nan,
        "best_shift_start_diff": np.nan,
        "best_shift_end_diff": np.nan,
        "best_shift_duration_diff": np.nan,
        "second_shift_candidate": np.nan,
        "second_shift_score": np.nan,
        "candidate_score_gap": np.nan,
        "pair_first_punch": pd.NaT,
        "pair_last_punch": pd.NaT,
        "pair_start_min": np.nan,
        "pair_end_min": np.nan,
        "pair_span_min": np.nan,
        "pair_punch_count": 0,
        "pair_outside_punch_count": 0,
        "pair_outside_debug_count": 0,
        "pair_ignored_after_out_count": 0,
        "pair_max_gap_min": 0.0,
        "pair_mean_gap_min": 0.0,
        "pair_unique_reader_count": 0,
        "pair_unique_reason_count": 0,
        "pair_unique_txncode_count": 0,
        "pair_next_day_flag": 0,
        "pair_preview": "",
        "raw_transactions": "",
        "pair_no_punch_flag": 1,
        "pair_single_punch_flag": 0,
        "pair_overtime_like_flag": 0,
        "pair_extended_duty_flag": 0,
        "pair_long_duty_flag": 0,
        "pair_manual_continuous_flag": 0,
        "pair_confidence_bucket": "unknown",
        "reader_pair_method": "",
        "reader_pair_out_scope": "",
    }
    for _, _, name in TIME_BUCKETS:
        defaults[name] = 0
    for column, value in defaults.items():
        df[column] = value
    for column in (
        "best_shift_candidate",
        "second_shift_candidate",
        "pair_preview",
        "raw_transactions",
        "pair_confidence_bucket",
        "reader_pair_method",
        "reader_pair_out_scope",
    ):
        df[column] = df[column].astype(object)
    return df


def _expand_transactions_to_windows(txn):
    if txn.empty:
        return pd.DataFrame(columns=[
            "PersonCode_norm",
            "attendance_day",
            "ts",
            "reader_number",
            "reader",
            "reason",
            "txn_code",
            "minute_from_day0",
        ])

    base = pd.DataFrame({
        "PersonCode_norm": txn["EmpCode_norm"].astype(str),
        "txn_day": txn["txn_day"],
        "ts": txn["TransactionDateTime"],
        "reader_number": txn["ReaderNumber"],
        "reader": txn["ReaderId"],
        "reason": txn["ReasonCode"],
        "txn_code": txn["TransactionCode"],
    })

    current = base.rename(columns={"txn_day": "attendance_day"})
    previous = base.copy()
    previous["attendance_day"] = previous["txn_day"] - pd.Timedelta(days=1)
    previous = previous.drop(columns=["txn_day"])

    expanded = pd.concat([current, previous], ignore_index=True)
    expanded["minute_from_day0"] = (
        (expanded["ts"] - expanded["attendance_day"]).dt.total_seconds() / 60.0
    )
    expanded = expanded[
        (expanded["minute_from_day0"] >= 0) &
        (expanded["minute_from_day0"] < WINDOW_HOURS_AFTER * 60)
    ].copy()
    expanded = expanded.sort_values(
        ["PersonCode_norm", "attendance_day", "ts"],
        kind="stable",
    ).reset_index(drop=True)
    return expanded


def _window_gap_stats(expanded):
    if expanded.empty:
        return pd.DataFrame(columns=[
            "PersonCode_norm",
            "attendance_day",
            "window_large_gap_count",
            "window_cluster_count",
        ])

    keyed = expanded[["PersonCode_norm", "attendance_day", "ts"]].copy()
    keyed["_gap_min"] = (
        keyed.groupby(["PersonCode_norm", "attendance_day"], sort=False)["ts"]
        .diff()
        .dt.total_seconds()
        .floordiv(60)
    )
    keyed["_large_gap"] = keyed["_gap_min"].gt(240).astype(int)
    gaps = (
        keyed.groupby(["PersonCode_norm", "attendance_day"], as_index=False, sort=False)
        .agg(window_large_gap_count=("_large_gap", "sum"))
    )
    gaps["window_cluster_count"] = gaps["window_large_gap_count"] + 1
    return gaps


def _build_window_summary(expanded):
    if expanded.empty:
        return pd.DataFrame(columns=["PersonCode_norm", "attendance_day"])

    summary = (
        expanded.groupby(["PersonCode_norm", "attendance_day"], as_index=False, sort=False)
        .agg(
            window_punch_count=("ts", "size"),
            reader_in_count=("reader_number", lambda s: int((s == 1).sum())),
            reader_out_count=("reader_number", lambda s: int((s == 2).sum())),
        )
    )
    gap_stats = _window_gap_stats(expanded)
    summary = summary.merge(gap_stats, on=["PersonCode_norm", "attendance_day"], how="left")
    summary["window_large_gap_count"] = summary["window_large_gap_count"].fillna(0).astype(int)
    summary["window_cluster_count"] = np.where(
        summary["window_punch_count"].gt(0),
        summary["window_large_gap_count"] + 1,
        0,
    ).astype(int)

    raw_preview = (
        expanded.groupby(["PersonCode_norm", "attendance_day"], sort=False)["ts"]
        .agg(lambda s: _join_limited_timestamps(s, 12))
        .reset_index(name="raw_transactions")
    )
    return summary.merge(raw_preview, on=["PersonCode_norm", "attendance_day"], how="left")


def _build_same_day_pairs(txn):
    if txn.empty:
        return pd.DataFrame(columns=["PersonCode_norm", "attendance_day"])

    in_rows = txn[txn["ReaderNumber"].eq(1)]
    if in_rows.empty:
        return pd.DataFrame(columns=["PersonCode_norm", "attendance_day"])

    same_day_in = (
        in_rows.groupby(["EmpCode_norm", "txn_day"], as_index=False, sort=False)
        .agg(pair_first_punch=("TransactionDateTime", "min"))
        .rename(columns={"EmpCode_norm": "PersonCode_norm", "txn_day": "attendance_day"})
    )

    out_rows = txn[txn["ReaderNumber"].eq(2)][["EmpCode_norm", "txn_day", "TransactionDateTime"]]
    if out_rows.empty:
        same_day_in["pair_last_punch"] = pd.NaT
        return same_day_in

    out_candidates = out_rows.merge(
        same_day_in,
        left_on=["EmpCode_norm", "txn_day"],
        right_on=["PersonCode_norm", "attendance_day"],
        how="inner",
    )
    out_candidates = out_candidates[out_candidates["TransactionDateTime"] > out_candidates["pair_first_punch"]]
    same_day_out = (
        out_candidates.groupby(["PersonCode_norm", "attendance_day"], as_index=False, sort=False)
        .agg(pair_last_punch=("TransactionDateTime", "max"))
    )
    return same_day_in.merge(same_day_out, on=["PersonCode_norm", "attendance_day"], how="left")


def _add_fallback_outs(pair_df, txn):
    if pair_df.empty or txn.empty:
        return pair_df

    result = pair_df.copy()
    missing = result["pair_first_punch"].notna() & result["pair_last_punch"].isna()
    if not missing.any():
        result["reader_pair_out_scope"] = "same_day"
        result["reader_pair_method"] = "day_first_earliest_reader1_same_day_latest_reader2_after_in"
        return result

    out_rows = txn[txn["ReaderNumber"].eq(2)][["EmpCode_norm", "TransactionDateTime"]].copy()
    out_rows = out_rows.sort_values(["EmpCode_norm", "TransactionDateTime"], kind="stable")

    fallback_rows = result.loc[missing, ["PersonCode_norm", "attendance_day", "pair_first_punch"]].copy()
    fallback_rows["_row_id"] = fallback_rows.index
    fallback_candidates = fallback_rows.merge(
        out_rows,
        left_on="PersonCode_norm",
        right_on="EmpCode_norm",
        how="left",
    )
    fallback_candidates = fallback_candidates[
        (fallback_candidates["TransactionDateTime"] > fallback_candidates["pair_first_punch"]) &
        (fallback_candidates["TransactionDateTime"] <= fallback_candidates["pair_first_punch"] + pd.Timedelta(minutes=MAX_LONG_DUTY_SPAN))
    ].copy()

    result["reader_pair_out_scope"] = np.where(
        result["pair_last_punch"].notna(),
        "same_day",
        "",
    )
    result["reader_pair_method"] = np.where(
        result["pair_last_punch"].notna(),
        "day_first_earliest_reader1_same_day_latest_reader2_after_in",
        "",
    )

    if fallback_candidates.empty:
        return result

    fallback_candidates["_minutes_after_in"] = (
        (fallback_candidates["TransactionDateTime"] - fallback_candidates["pair_first_punch"])
        .dt.total_seconds()
        .div(60.0)
    )

    selected_parts = []
    for max_minutes, scope, method in (
        (MAX_NORMAL_SPAN, "normal_overnight_16h", "day_first_earliest_reader1_fallback_latest_reader2_within_16h_after_in"),
        (MAX_EXTENDED_SPAN, "extended_24h", "day_first_earliest_reader1_extended_latest_reader2_within_24h_after_in"),
        (MAX_LONG_DUTY_SPAN, "long_duty_48h", "day_first_earliest_reader1_long_duty_latest_reader2_within_48h_after_in"),
    ):
        remaining_ids = set(fallback_rows["_row_id"]) - set(pd.concat(selected_parts)["_row_id"]) if selected_parts else set(fallback_rows["_row_id"])
        if not remaining_ids:
            break
        scoped = fallback_candidates[
            fallback_candidates["_row_id"].isin(remaining_ids) &
            fallback_candidates["_minutes_after_in"].le(max_minutes)
        ]
        if scoped.empty:
            continue
        selected = (
            scoped.groupby("_row_id", as_index=False, sort=False)
            .agg(pair_last_punch=("TransactionDateTime", "max"))
        )
        selected["reader_pair_out_scope"] = scope
        selected["reader_pair_method"] = method
        selected_parts.append(selected)

    if not selected_parts:
        return result

    selected_all = pd.concat(selected_parts, ignore_index=True).drop_duplicates("_row_id", keep="first")
    result.loc[selected_all["_row_id"], "pair_last_punch"] = selected_all["pair_last_punch"].to_numpy()
    result.loc[selected_all["_row_id"], "reader_pair_out_scope"] = selected_all["reader_pair_out_scope"].to_numpy()
    result.loc[selected_all["_row_id"], "reader_pair_method"] = selected_all["reader_pair_method"].to_numpy()
    return result


def _score_pairs(pair_df):
    if pair_df.empty:
        return pair_df

    result = pair_df.copy()
    found = result["pair_first_punch"].notna() & result["pair_last_punch"].notna()
    if not found.any():
        return result

    scores = []
    for shift_name in MODELED_SHIFT_CLASSES:
        meta = SHIFT_META[shift_name]
        start_diff = (result["pair_start_min"] - float(meta["start_min"])).abs()
        end_diff = (result["pair_end_min"] - float(meta["end_min"])).abs()
        duration_diff = (result["pair_span_min"] - float(meta["duration_min"])).abs()

        penalty = result["pair_outside_punch_count"].fillna(0).astype(float) * 10.0
        penalty += np.where(result["pair_punch_count"].fillna(0).lt(2), 300.0, 0.0)
        penalty += np.where(result["pair_span_min"].lt(MIN_VALID_SPAN), 400.0, 0.0)
        penalty += np.where(result["pair_span_min"].gt(MAX_NORMAL_SPAN), 250.0, 0.0)
        penalty += np.where(result["pair_span_min"].gt(MAX_EXTENDED_SPAN), 500.0, 0.0)
        penalty += np.where(result["pair_span_min"].gt(MAX_LONG_DUTY_SPAN), 1500.0, 0.0)

        reward = np.zeros(len(result), dtype=float)
        reward += np.where(result["pair_punch_count"].fillna(0).ge(3), 15.0, 0.0)
        reward += np.where(start_diff.le(30), 25.0, np.where(start_diff.le(60), 12.0, 0.0))
        reward += np.where(end_diff.le(30), 18.0, np.where(end_diff.le(60), 8.0, 0.0))
        reward += np.where(duration_diff.le(45), 20.0, np.where(duration_diff.le(90), 8.0, 0.0))

        score = start_diff * 0.90 + end_diff * 0.75 + duration_diff * 0.65 + penalty - reward
        scores.append(pd.DataFrame({
            "shift": shift_name,
            "score": score,
            "start_diff": start_diff,
            "end_diff": end_diff,
            "duration_diff": duration_diff,
        }, index=result.index))

    score_df = pd.concat(scores, ignore_index=False)
    score_df = score_df.loc[found[found].index]
    score_df["_source_index"] = score_df.index
    ranked = score_df.sort_values(
        ["_source_index", "score", "duration_diff", "start_diff", "end_diff"],
        kind="stable",
    )
    best = ranked.groupby("_source_index", sort=False).nth(0)
    second = ranked.groupby("_source_index", sort=False).nth(1)

    result.loc[best.index, "best_shift_candidate"] = best["shift"].to_numpy()
    result.loc[best.index, "best_shift_score"] = best["score"].to_numpy()
    result.loc[best.index, "best_shift_start_diff"] = best["start_diff"].to_numpy()
    result.loc[best.index, "best_shift_end_diff"] = best["end_diff"].to_numpy()
    result.loc[best.index, "best_shift_duration_diff"] = best["duration_diff"].to_numpy()

    if not second.empty:
        result.loc[second.index, "second_shift_candidate"] = second["shift"].to_numpy()
        result.loc[second.index, "second_shift_score"] = second["score"].to_numpy()
        result.loc[second.index, "candidate_score_gap"] = (
            second["score"] - result.loc[second.index, "best_shift_score"]
        ).to_numpy()

    return result


def _build_fast_features(txn, base):
    feature_df = _empty_fast_feature_frame(base.index)
    if base.empty:
        return feature_df

    keys = base[["PersonCode_norm", "attendance_day"]].copy()
    keys["_base_index"] = base.index
    expanded = _expand_transactions_to_windows(txn)
    window_summary = _build_window_summary(expanded)
    feature_df = feature_df.reset_index(drop=True)

    if not window_summary.empty:
        keyed_window = keys.merge(window_summary, on=["PersonCode_norm", "attendance_day"], how="left")
        for column in (
            "window_punch_count",
            "reader_in_count",
            "reader_out_count",
            "window_large_gap_count",
            "window_cluster_count",
        ):
            feature_df[column] = keyed_window[column].fillna(0).astype(int)
        feature_df["raw_transactions"] = keyed_window["raw_transactions"].fillna("")

    feature_df["pair_no_punch_flag"] = feature_df["window_punch_count"].eq(0).astype(int)
    feature_df["pair_single_punch_flag"] = feature_df["window_punch_count"].eq(1).astype(int)

    pairs = _build_same_day_pairs(txn)
    pairs = _add_fallback_outs(pairs, txn)
    if pairs.empty:
        return feature_df

    pairs = keys.merge(pairs, on=["PersonCode_norm", "attendance_day"], how="left")
    found = pairs["pair_first_punch"].notna() & pairs["pair_last_punch"].notna()
    positive_span = (
        (pairs["pair_last_punch"] - pairs["pair_first_punch"]).dt.total_seconds().div(60.0) > 0
    )
    found &= positive_span.fillna(False)
    if not found.any():
        return feature_df

    pairs["pair_start_min"] = (
        (pairs["pair_first_punch"] - pairs["attendance_day"]).dt.total_seconds() / 60.0
    )
    pairs["pair_end_min"] = (
        (pairs["pair_last_punch"] - pairs["attendance_day"]).dt.total_seconds() / 60.0
    )
    pairs["pair_span_min"] = pairs["pair_end_min"] - pairs["pair_start_min"]
    pairs["reader_pair_found_flag"] = found.astype(int)
    pairs["pair_next_day_flag"] = (
        pairs["pair_last_punch"].dt.normalize() > pairs["attendance_day"].dt.normalize()
    ).fillna(False).astype(int)

    pair_bounds = pairs.loc[found, [
        "_base_index",
        "PersonCode_norm",
        "attendance_day",
        "pair_first_punch",
        "pair_last_punch",
        "reader_pair_out_scope",
    ]].copy()
    pair_map = expanded.merge(pair_bounds, on=["PersonCode_norm", "attendance_day"], how="inner")
    pair_map = pair_map[
        (pair_map["ts"] >= pair_map["pair_first_punch"]) &
        (pair_map["ts"] <= pair_map["pair_last_punch"])
    ].copy()

    if not pair_map.empty:
        pair_map["_gap_min"] = (
            pair_map.sort_values(["_base_index", "ts"], kind="stable")
            .groupby("_base_index", sort=False)["ts"]
            .diff()
            .dt.total_seconds()
            .floordiv(60)
        )
        pair_summary = (
            pair_map.groupby("_base_index", as_index=True, sort=False)
            .agg(
                pair_punch_count=("ts", "size"),
                pair_max_gap_min=("_gap_min", "max"),
                pair_mean_gap_min=("_gap_min", "mean"),
                pair_unique_reader_count=("reader", "nunique"),
                pair_unique_reason_count=("reason", "nunique"),
                pair_unique_txncode_count=("txn_code", "nunique"),
                pair_preview=("ts", lambda s: _join_limited_timestamps(s, 12)),
            )
        )
        pair_summary[["pair_max_gap_min", "pair_mean_gap_min"]] = pair_summary[["pair_max_gap_min", "pair_mean_gap_min"]].fillna(0.0)

        for column in pair_summary.columns:
            feature_df.loc[pair_summary.index, column] = pair_summary[column]

        pair_map["minute_from_day0"] = pair_map["minute_from_day0"].astype(float)
        for start_min, end_min, name in TIME_BUCKETS:
            counts = (
                pair_map[
                    (pair_map["minute_from_day0"] >= start_min) &
                    (pair_map["minute_from_day0"] < end_min)
                ]
                .groupby("_base_index", sort=False)
                .size()
            )
            feature_df.loc[counts.index, name] = counts.astype(int)

    found_index = pairs.index[found]
    feature_df.loc[found_index, "reader_pair_found_flag"] = 1
    for column in ("pair_first_punch", "pair_last_punch", "pair_start_min", "pair_end_min", "pair_span_min", "pair_next_day_flag"):
        feature_df.loc[found_index, column] = pairs.loc[found_index, column].to_numpy()

    feature_df.loc[found_index, "reader_pair_method"] = pairs.loc[found_index, "reader_pair_method"].fillna("").to_numpy()
    feature_df.loc[found_index, "reader_pair_out_scope"] = pairs.loc[found_index, "reader_pair_out_scope"].fillna("").to_numpy()

    feature_df.loc[found_index, "pair_outside_debug_count"] = (
        feature_df.loc[found_index, "window_punch_count"].astype(int) -
        feature_df.loc[found_index, "pair_punch_count"].astype(int)
    ).clip(lower=0)

    out_time = pd.to_datetime(feature_df.loc[found_index, "pair_last_punch"])
    in_time = pd.to_datetime(feature_df.loc[found_index, "pair_first_punch"])
    same_day_scope = feature_df.loc[found_index, "reader_pair_out_scope"].eq("same_day")

    same_day_counts = base.loc[found_index, "txn_punch_count"].fillna(0).astype(int)
    pair_counts = feature_df.loc[found_index, "pair_punch_count"].fillna(0).astype(int)
    outside_same_day = (same_day_counts - pair_counts).clip(lower=0)

    before_or_at_out = []
    after_out = []
    before_in = []
    expanded_by_key = {
        key: group["ts"].to_numpy(dtype="datetime64[ns]")
        for key, group in expanded.groupby(["PersonCode_norm", "attendance_day"], sort=False)
    }
    for idx in found_index:
        key = (pairs.at[idx, "PersonCode_norm"], pairs.at[idx, "attendance_day"])
        arr = expanded_by_key.get(key)
        if arr is None:
            before_or_at_out.append(0)
            after_out.append(0)
            before_in.append(0)
            continue
        out_np = np.datetime64(out_time.loc[idx])
        in_np = np.datetime64(in_time.loc[idx])
        before_or_at_out.append(int(np.searchsorted(arr, out_np, side="right")))
        after_out.append(int(len(arr) - np.searchsorted(arr, out_np, side="right")))
        before_in.append(int(np.searchsorted(arr, in_np, side="left")))

    before_or_at_out = pd.Series(before_or_at_out, index=found_index)
    before_in = pd.Series(before_in, index=found_index)
    after_out = pd.Series(after_out, index=found_index)

    feature_df.loc[found_index, "pair_ignored_after_out_count"] = after_out
    feature_df.loc[found_index, "pair_outside_punch_count"] = np.where(
        same_day_scope.to_numpy(),
        outside_same_day.to_numpy(),
        (before_or_at_out - pair_counts).clip(lower=0).to_numpy(),
    )

    pairs_for_score = feature_df.loc[found_index].copy()
    scored = _score_pairs(pairs_for_score)
    for column in (
        "best_shift_candidate",
        "best_shift_score",
        "best_shift_start_diff",
        "best_shift_end_diff",
        "best_shift_duration_diff",
        "second_shift_candidate",
        "second_shift_score",
        "candidate_score_gap",
    ):
        feature_df.loc[found_index, column] = scored[column]

    chosen_shift = feature_df.loc[found_index, "best_shift_candidate"]
    expected = chosen_shift.map(lambda shift: float(SHIFT_META.get(str(shift), {}).get("duration_min", np.nan)))
    span = feature_df.loc[found_index, "pair_span_min"].astype(float)
    start_diff = feature_df.loc[found_index, "best_shift_start_diff"].astype(float)
    end_diff = feature_df.loc[found_index, "best_shift_end_diff"].astype(float)
    dur_diff = feature_df.loc[found_index, "best_shift_duration_diff"].astype(float)
    score = feature_df.loc[found_index, "best_shift_score"].astype(float)
    gap = feature_df.loc[found_index, "candidate_score_gap"].astype(float)

    feature_df.loc[found_index, "pair_overtime_like_flag"] = (
        span.ge(expected + 120) & span.le(expected + 360) & start_diff.le(120)
    ).fillna(False).astype(int)
    feature_df.loc[found_index, "pair_extended_duty_flag"] = (
        span.gt(MAX_NORMAL_SPAN) & span.le(MAX_EXTENDED_SPAN)
    ).astype(int)
    feature_df.loc[found_index, "pair_long_duty_flag"] = (
        span.gt(MAX_EXTENDED_SPAN) & span.le(MAX_LONG_DUTY_SPAN)
    ).astype(int)

    method = feature_df.loc[found_index, "reader_pair_method"].astype(str)
    pair_max_gap = feature_df.loc[found_index, "pair_max_gap_min"].astype(float)
    feature_df.loc[found_index, "pair_manual_continuous_flag"] = (
        feature_df.loc[found_index, "pair_extended_duty_flag"].eq(1) |
        feature_df.loc[found_index, "pair_long_duty_flag"].eq(1) |
        method.str.contains("within_24h", regex=False) |
        method.str.contains("within_48h", regex=False) |
        span.gt(expected + 360) |
        pair_max_gap.gt(720)
    ).fillna(False).astype(int)

    very_high = gap.notna() & score.le(80) & gap.ge(50) & start_diff.le(60) & end_diff.le(90) & dur_diff.le(90)
    high = gap.notna() & score.le(120) & gap.ge(30)
    medium = score.le(180)
    bucket = pd.Series("low", index=found_index)
    bucket.loc[medium] = "medium"
    bucket.loc[high] = "high"
    bucket.loc[very_high] = "very_high"
    feature_df.loc[found_index, "pair_confidence_bucket"] = bucket

    return feature_df


def _pair_score(start_min, end_min, duration_min, shift_name, pair_punch_count, outside_punch_count):
    meta = SHIFT_META[shift_name]

    start_diff = abs(start_min - float(meta["start_min"]))
    end_diff = abs(end_min - float(meta["end_min"]))
    duration_diff = abs(duration_min - float(meta["duration_min"]))

    penalty = 0.0
    penalty += outside_punch_count * 10.0

    if pair_punch_count < 2:
        penalty += 300.0

    if duration_min < MIN_VALID_SPAN:
        penalty += 400.0

    if duration_min > MAX_NORMAL_SPAN:
        penalty += 250.0
    if duration_min > MAX_EXTENDED_SPAN:
        penalty += 500.0
    if duration_min > MAX_LONG_DUTY_SPAN:
        penalty += 1500.0

    reward = 0.0

    if pair_punch_count >= 3:
        reward += 15.0

    if start_diff <= 30:
        reward += 25.0
    elif start_diff <= 60:
        reward += 12.0

    if end_diff <= 30:
        reward += 18.0
    elif end_diff <= 60:
        reward += 8.0

    if duration_diff <= 45:
        reward += 20.0
    elif duration_diff <= 90:
        reward += 8.0

    score = (
        start_diff * 0.90 +
        end_diff * 0.75 +
        duration_diff * 0.65 +
        penalty -
        reward
    )

    return {
        "score": float(score),
        "start_diff": float(start_diff),
        "end_diff": float(end_diff),
        "duration_diff": float(duration_diff),
    }


def _build_fixed_reader_pair(win_df):
    if win_df.empty:
        return None

    # Punch In = earliest same-day ReaderNumber 1
    in_df = win_df[
        (win_df["reader_number"] == 1) &
        (win_df["minute_from_day0"] >= 0) &
        (win_df["minute_from_day0"] < 1440)
    ].copy()

    if in_df.empty:
        return None

    in_row = in_df.sort_values("ts").iloc[0]
    in_time = pd.Timestamp(in_row["ts"])
    in_pos = int(in_row["pos"])
    in_min = float(in_row["minute_from_day0"])

    # Priority 1: latest same-day OUT after IN
    same_day_out_df = win_df[
        (win_df["reader_number"] == 2) &
        (win_df["ts"] > in_time) &
        (win_df["minute_from_day0"] >= 0) &
        (win_df["minute_from_day0"] < 1440)
    ].copy()

    if not same_day_out_df.empty:
        out_row = same_day_out_df.sort_values("ts").iloc[-1]
        out_method = "same_day_latest_reader2_after_in"
        out_scope = "same_day"
    else:
        minutes_after_in = (win_df["ts"] - in_time).dt.total_seconds() / 60.0

        out_16h = win_df[
            (win_df["reader_number"] == 2) &
            (win_df["ts"] > in_time) &
            (minutes_after_in <= MAX_NORMAL_SPAN)
        ].copy()

        if not out_16h.empty:
            out_row = out_16h.sort_values("ts").iloc[-1]
            out_method = "fallback_latest_reader2_within_16h_after_in"
            out_scope = "normal_overnight_16h"
        else:
            out_24h = win_df[
                (win_df["reader_number"] == 2) &
                (win_df["ts"] > in_time) &
                (minutes_after_in <= MAX_EXTENDED_SPAN)
            ].copy()

            if not out_24h.empty:
                out_row = out_24h.sort_values("ts").iloc[-1]
                out_method = "extended_latest_reader2_within_24h_after_in"
                out_scope = "extended_24h"
            else:
                out_48h = win_df[
                    (win_df["reader_number"] == 2) &
                    (win_df["ts"] > in_time) &
                    (minutes_after_in <= MAX_LONG_DUTY_SPAN)
                ].copy()

                if out_48h.empty:
                    return None

                out_row = out_48h.sort_values("ts").iloc[-1]
                out_method = "long_duty_latest_reader2_within_48h_after_in"
                out_scope = "long_duty_48h"

    out_time = pd.Timestamp(out_row["ts"])
    out_pos = int(out_row["pos"])
    out_min = float(out_row["minute_from_day0"])
    span = out_min - in_min

    if span <= 0:
        return None

    pair_df = win_df[
        (win_df["pos"] >= in_pos) &
        (win_df["pos"] <= out_pos)
    ].copy().reset_index(drop=True)

    outside_debug_count = int(len(win_df) - len(pair_df))

    if out_scope == "same_day":
        score_scope_df = win_df[
            (win_df["minute_from_day0"] >= 0) &
            (win_df["minute_from_day0"] < 1440)
        ].copy()
    else:
        score_scope_df = win_df[win_df["ts"] <= out_time].copy()

    score_pair_positions = set(win_df.loc[
        (win_df["pos"] >= in_pos) &
        (win_df["pos"] <= out_pos),
        "pos"
    ].tolist())

    outside_score_count = int(
        (~score_scope_df["pos"].isin(score_pair_positions)).sum()
    )

    ignored_after_out_count = int((win_df["ts"] > out_time).sum())

    return {
        "in_time": in_time,
        "out_time": out_time,
        "in_pos": in_pos,
        "out_pos": out_pos,
        "in_min": in_min,
        "out_min": out_min,
        "span": span,
        "pair_df": pair_df,
        "pair_punch_count": int(len(pair_df)),
        "outside_punch_count": int(outside_score_count),
        "outside_debug_count": int(outside_debug_count),
        "ignored_after_out_count": int(ignored_after_out_count),
        "out_scope": out_scope,
        "reader_pair_method": f"day_first_earliest_reader1_{out_method}",
    }


def _choose_best_fixed_reader_candidate(win_df):
    fixed_pair = _build_fixed_reader_pair(win_df)

    if fixed_pair is None:
        return None, None, []

    candidates = []

    for sh in MODELED_SHIFT_CLASSES:
        if sh not in SHIFT_META:
            continue

        scored = _pair_score(
            start_min=fixed_pair["in_min"],
            end_min=fixed_pair["out_min"],
            duration_min=fixed_pair["span"],
            shift_name=sh,
            pair_punch_count=fixed_pair["pair_punch_count"],
            outside_punch_count=fixed_pair["outside_punch_count"],
        )

        candidates.append({
            "shift": sh,
            "score": scored["score"],
            "start_diff": scored["start_diff"],
            "end_diff": scored["end_diff"],
            "duration_diff": scored["duration_diff"],
            "i": fixed_pair["in_pos"],
            "j": fixed_pair["out_pos"],
            "pair_punch_count": fixed_pair["pair_punch_count"],
            "outside_punch_count": fixed_pair["outside_punch_count"],
            "outside_debug_count": fixed_pair["outside_debug_count"],
            "ignored_after_out_count": fixed_pair["ignored_after_out_count"],
            "out_scope": fixed_pair["out_scope"],
            "reader_in_time": fixed_pair["in_time"],
            "reader_out_time": fixed_pair["out_time"],
            "fixed_pair_span": fixed_pair["span"],
            "reader_pair_method": fixed_pair["reader_pair_method"],
        })

    if not candidates:
        return None, None, []

    ranked = sorted(
        candidates,
        key=lambda x: (
            x["score"],
            x["duration_diff"] if pd.notna(x["duration_diff"]) else 999999,
            x["start_diff"],
            x["end_diff"],
        )
    )

    return ranked[0], ranked[1] if len(ranked) > 1 else None, ranked


def _extract_fixed_reader_features(win_df, day0, best, second):
    base = {
        "window_punch_count": int(len(win_df)),
        "window_large_gap_count": 0,
        "window_cluster_count": 0,

        "reader_in_count": 0,
        "reader_out_count": 0,
        "reader_pair_found_flag": 0,

        "best_shift_candidate": np.nan,
        "best_shift_score": np.nan,
        "best_shift_start_diff": np.nan,
        "best_shift_end_diff": np.nan,
        "best_shift_duration_diff": np.nan,

        "second_shift_candidate": np.nan,
        "second_shift_score": np.nan,
        "candidate_score_gap": np.nan,

        "pair_first_punch": pd.NaT,
        "pair_last_punch": pd.NaT,
        "pair_start_min": np.nan,
        "pair_end_min": np.nan,
        "pair_span_min": np.nan,
        "pair_punch_count": 0,
        "pair_outside_punch_count": 0,
        "pair_outside_debug_count": 0,
        "pair_ignored_after_out_count": 0,
        "pair_max_gap_min": 0.0,
        "pair_mean_gap_min": 0.0,
        "pair_unique_reader_count": 0,
        "pair_unique_reason_count": 0,
        "pair_unique_txncode_count": 0,
        "pair_next_day_flag": 0,
        "pair_preview": "",
        "raw_transactions": "",

        "pair_no_punch_flag": 1,
        "pair_single_punch_flag": 0,
        "pair_overtime_like_flag": 0,
        "pair_extended_duty_flag": 0,
        "pair_long_duty_flag": 0,
        "pair_manual_continuous_flag": 0,
        "pair_confidence_bucket": "unknown",
        "reader_pair_method": "",
        "reader_pair_out_scope": "",
    }

    for _, _, name in TIME_BUCKETS:
        base[name] = 0

    if win_df.empty:
        return base

    full_ts = pd.to_datetime(win_df["ts"]).sort_values().reset_index(drop=True)
    base["raw_transactions"] = " | ".join(full_ts.astype(str).tolist()[:30])

    base["reader_in_count"] = int((win_df["reader_number"] == 1).sum())
    base["reader_out_count"] = int((win_df["reader_number"] == 2).sum())

    _, _, large_gap_count = _build_gap_stats(full_ts)

    base["window_large_gap_count"] = int(large_gap_count)
    base["window_cluster_count"] = int(large_gap_count + 1) if len(full_ts) > 0 else 0
    base["pair_no_punch_flag"] = 0

    if len(full_ts) == 1:
        base["pair_single_punch_flag"] = 1

    if best is None:
        return base

    base["reader_pair_found_flag"] = 1

    base["best_shift_candidate"] = best["shift"]
    base["best_shift_score"] = float(best["score"])
    base["best_shift_start_diff"] = float(best["start_diff"])
    base["best_shift_end_diff"] = float(best["end_diff"])
    base["best_shift_duration_diff"] = float(best["duration_diff"])

    if second is not None:
        base["second_shift_candidate"] = second["shift"]
        base["second_shift_score"] = float(second["score"])
        base["candidate_score_gap"] = float(second["score"] - best["score"])

    pair_df = win_df[
        (win_df["pos"] >= int(best["i"])) &
        (win_df["pos"] <= int(best["j"]))
    ].copy().reset_index(drop=True)

    pair_ts = pd.to_datetime(pair_df["ts"]).sort_values().reset_index(drop=True)
    pair_mins = pair_ts.apply(lambda x: _minutes_from_day0(x, day0)).astype(float)

    base["pair_first_punch"] = pd.Timestamp(best["reader_in_time"])
    base["pair_last_punch"] = pd.Timestamp(best["reader_out_time"])
    base["pair_start_min"] = float(_minutes_from_day0(base["pair_first_punch"], day0))
    base["pair_end_min"] = float(_minutes_from_day0(base["pair_last_punch"], day0))
    base["pair_span_min"] = float(base["pair_end_min"] - base["pair_start_min"])

    base["pair_punch_count"] = int(len(pair_df))
    base["pair_outside_punch_count"] = int(best.get("outside_punch_count", 0))
    base["pair_outside_debug_count"] = int(best.get("outside_debug_count", 0))
    base["pair_ignored_after_out_count"] = int(best.get("ignored_after_out_count", 0))
    base["reader_pair_out_scope"] = best.get("out_scope", "")

    base["pair_next_day_flag"] = int(
        pd.Timestamp(base["pair_last_punch"]).normalize() > pd.Timestamp(day0).normalize()
    )

    base["pair_preview"] = " | ".join(pair_ts.astype(str).tolist()[:12])
    base["reader_pair_method"] = best.get("reader_pair_method", "")

    max_gap, mean_gap, _ = _build_gap_stats(pair_ts)

    base["pair_max_gap_min"] = float(max_gap)
    base["pair_mean_gap_min"] = float(mean_gap)

    base["pair_unique_reader_count"] = int(pd.Series(pair_df["reader"]).nunique(dropna=True))
    base["pair_unique_reason_count"] = int(pd.Series(pair_df["reason"]).nunique(dropna=True))
    base["pair_unique_txncode_count"] = int(pd.Series(pair_df["txn_code"]).nunique(dropna=True))

    for k, v in _bucket_counts(pair_mins).items():
        base[k] = v

    chosen_shift = best["shift"]
    expected = float(SHIFT_META[chosen_shift]["duration_min"])
    span = base["pair_span_min"]

    base["pair_overtime_like_flag"] = int(
        span >= expected + 120 and
        span <= expected + 360 and
        base["best_shift_start_diff"] <= 120
    )

    base["pair_extended_duty_flag"] = int(span > MAX_NORMAL_SPAN and span <= MAX_EXTENDED_SPAN)
    base["pair_long_duty_flag"] = int(span > MAX_EXTENDED_SPAN and span <= MAX_LONG_DUTY_SPAN)

    base["pair_manual_continuous_flag"] = int(
        (base["pair_extended_duty_flag"] == 1) or
        (base["pair_long_duty_flag"] == 1) or
        ("within_24h" in base["reader_pair_method"]) or
        ("within_48h" in base["reader_pair_method"]) or
        (span > expected + 360) or
        (base["pair_max_gap_min"] > 720)
    )

    score = base["best_shift_score"]
    gap = base["candidate_score_gap"]
    start_diff = base["best_shift_start_diff"]
    end_diff = base["best_shift_end_diff"]
    dur_diff = base["best_shift_duration_diff"]

    if pd.notna(gap) and score <= 80 and gap >= 50 and start_diff <= 60 and end_diff <= 90 and dur_diff <= 90:
        base["pair_confidence_bucket"] = "very_high"
    elif pd.notna(gap) and score <= 120 and gap >= 30:
        base["pair_confidence_bucket"] = "high"
    elif score <= 180:
        base["pair_confidence_bucket"] = "medium"
    else:
        base["pair_confidence_bucket"] = "low"

    return base


def _safe_int(row, col, default=0):
    try:
        return int(row.get(col, default) or default)
    except Exception:
        return default


def _safe_float(row, col, default=np.nan):
    try:
        val = row.get(col, default)
        return float(val) if pd.notna(val) else default
    except Exception:
        return default


def _valid_model_shift(x):
    return str(x) in MODELED_SHIFT_CLASSES


def _final_decision(row):
    model_pred = row.get("prod_model_pred_shift", np.nan)
    conf = _safe_float(row, "prod_model_confidence", 0)

    pair_pred = row.get("best_shift_candidate", np.nan)
    pair_bucket = str(row.get("pair_confidence_bucket", "unknown"))

    same_day_txn_count = _safe_int(row, "txn_punch_count", 0)

    single_punch = _safe_int(row, "pair_single_punch_flag", 0)
    reader_pair_found = _safe_int(row, "reader_pair_found_flag", 0)

    manual_cont = _safe_int(row, "pair_manual_continuous_flag", 0)
    overtime_like = _safe_int(row, "pair_overtime_like_flag", 0)
    next_day = _safe_int(row, "pair_next_day_flag", 0)
    is_sunday = _safe_int(row, "is_sunday", 0)

    start_diff = _safe_float(row, "best_shift_start_diff", np.nan)
    end_diff = _safe_float(row, "best_shift_end_diff", np.nan)
    dur_diff = _safe_float(row, "best_shift_duration_diff", np.nan)
    score = _safe_float(row, "best_shift_score", np.nan)
    score_gap = _safe_float(row, "candidate_score_gap", np.nan)

    if same_day_txn_count == 0:
        return (
            "ABSENT",
            "ABSENT",
            "rule_generated",
            "ABSENT",
            "no_transaction_absent",
            "No transaction found; marked as absent"
        )

    if is_sunday == 1 and same_day_txn_count > 0:
        return (
            "WO",
            "WO",
            "review",
            "WO_REVIEW",
            "worked_on_sunday_weekoff_overtime",
            "Worked on weekly off / possible overtime or extra duty"
        )

    if reader_pair_found != 1:
        if single_punch == 1:
            return (
                "unknown",
                "WITHHELD",
                "withheld",
                "WITHHELD",
                "single_punch_no_valid_reader_pair",
                "Only one punch or no valid IN/OUT reader pair found"
            )

        return (
            "unknown",
            "WITHHELD",
            "withheld",
            "WITHHELD",
            "no_valid_reader_pair",
            "Transactions found but no valid ReaderNumber 1 IN and ReaderNumber 2 OUT pair"
        )

    if manual_cont == 1:
        return (
            model_pred,
            "WORKING",
            "review",
            "SHIFT_REVIEW",
            "manual_continuous_model_hint",
            "Continuous/manual-like punches found; model shift kept for review"
        )

    if conf >= 0.995 and _valid_model_shift(model_pred):
        return (
            model_pred,
            "WORKING",
            "predicted",
            "SHIFT_PREDICTED",
            "model_ultra_high",
            "High-confidence model prediction"
        )

    if conf >= 0.98 and model_pred == pair_pred and _valid_model_shift(model_pred):
        return (
            model_pred,
            "WORKING",
            "predicted",
            "SHIFT_PREDICTED",
            "model_pair_strong_agree",
            "Model and day-first reader-pair candidate strongly agree"
        )

    if (
        model_pred == pair_pred and
        next_day == 1 and
        pd.notna(start_diff) and start_diff <= 120 and
        pd.notna(dur_diff) and dur_diff <= 120 and
        conf >= 0.92 and
        _valid_model_shift(model_pred)
    ):
        return (
            model_pred,
            "WORKING",
            "predicted",
            "SHIFT_PREDICTED",
            "overnight_model_pair_anchor",
            "Overnight day-first reader-pair pattern agrees with model"
        )

    if (
        conf >= 0.95 and
        _valid_model_shift(model_pred) and
        (pd.isna(pair_pred) or pair_pred == model_pred or pair_bucket in {"medium", "unknown"})
    ):
        return (
            model_pred,
            "WORKING",
            "predicted",
            "SHIFT_PREDICTED",
            "model_high",
            "High-confidence model prediction"
        )

    if conf >= 0.85 and model_pred == pair_pred and _valid_model_shift(model_pred):
        return (
            model_pred,
            "WORKING",
            "review",
            "SHIFT_REVIEW",
            "model_pair_medium_agree",
            "Model and day-first reader-pair candidate agree, but review recommended"
        )

    if (
        pair_bucket == "very_high" and
        pd.notna(score) and score <= 90 and
        pd.notna(score_gap) and score_gap >= 40 and
        pd.notna(start_diff) and start_diff <= 90 and
        pd.notna(end_diff) and end_diff <= 120 and
        _valid_model_shift(pair_pred)
    ):
        return (
            pair_pred,
            "WORKING",
            "review",
            "SHIFT_REVIEW",
            "pair_strong_review",
            "Strong day-first reader-pair candidate found; review recommended"
        )

    if overtime_like == 1 and conf >= 0.80 and _valid_model_shift(model_pred):
        return (
            model_pred,
            "WORKING",
            "review",
            "SHIFT_REVIEW",
            "overtime_review",
            "Overtime-like day-first reader-pair span found; review recommended"
        )

    if conf >= 0.75 and _valid_model_shift(model_pred):
        return (
            model_pred,
            "WORKING",
            "review",
            "SHIFT_REVIEW",
            "low_conf_model_review",
            "Lower-confidence model prediction; review required"
        )

    return (
        "unknown",
        "WITHHELD",
        "withheld",
        "WITHHELD",
        "ambiguous",
        "Unable to confidently determine shift"
    )


def run_inference(input_file_path, output_dir=None, return_clean=True):
    timings = {}
    total_started = perf_counter()

    started = perf_counter()
    raw = _read_input_file(input_file_path)
    timings["file_read"] = perf_counter() - started

    started = perf_counter()
    txn = _normalize_columns(raw)
    timings["normalize"] = perf_counter() - started

    if txn.empty:
        raise ValueError("No valid transactions found after parsing input file.")

    started = perf_counter()
    calendar = _build_calendar(txn)
    timings["calendar_build"] = perf_counter() - started

    started = perf_counter()
    daily = _build_daily_summary(txn)
    timings["daily_summary"] = perf_counter() - started

    started = perf_counter()
    base = calendar.merge(
        daily,
        on=["PersonCode_norm", "attendance_day"],
        how="left"
    )

    base["txn_punch_count"] = pd.to_numeric(base["txn_punch_count"], errors="coerce").fillna(0).astype(int)
    base["txn_in_reader_count"] = pd.to_numeric(base["txn_in_reader_count"], errors="coerce").fillna(0).astype(int)
    base["txn_out_reader_count"] = pd.to_numeric(base["txn_out_reader_count"], errors="coerce").fillna(0).astype(int)

    base["attendance_weekday"] = base["attendance_day"].dt.weekday
    base["is_sunday"] = (base["attendance_weekday"] == 6).astype(int)

    started = perf_counter()
    feature_df = _build_fast_features(txn, base)
    timings["feature_generation"] = perf_counter() - started

    started = perf_counter()
    out = pd.concat(
        [base.reset_index(drop=True), feature_df.reset_index(drop=True)],
        axis=1
    )

    for c in FEATURE_COLS:
        if c not in out.columns:
            out[c] = -999
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(-999)

    X = out[FEATURE_COLS].copy()

    prob = MODEL.predict(X)
    best_idx = np.argmax(prob, axis=1)

    out["prod_model_pred_shift"] = [LABEL_CLASSES[i] for i in best_idx]
    out["prod_model_confidence"] = np.max(prob, axis=1)

    for i, cls in enumerate(LABEL_CLASSES):
        out[f"prod_prob_{cls}"] = prob[:, i]
    timings["model_prediction"] = perf_counter() - started

    started = perf_counter()
    decisions = out.apply(_final_decision, axis=1)

    out["final_shift"] = decisions.map(lambda x: x[0])
    out["final_day_status"] = decisions.map(lambda x: x[1])
    out["shift_status"] = decisions.map(lambda x: x[2])
    out["final_status_label"] = decisions.map(lambda x: x[3])
    out["decision_reason"] = decisions.map(lambda x: x[4])
    out["final_message"] = decisions.map(lambda x: x[5])
    timings["decision_layer"] = perf_counter() - started

    started = perf_counter()
    out["raw_transactions_debug"] = out.get("raw_transactions", "")
    out["pair_preview_debug"] = out.get("pair_preview", "")

    absent_mask = (
        (out["decision_reason"] == "no_transaction_absent") &
        (pd.to_numeric(out["txn_punch_count"], errors="coerce").fillna(0) == 0)
    )

    for c in ["raw_transactions", "pair_preview", "pair_first_punch", "pair_last_punch"]:
        if c in out.columns:
            out.loc[absent_mask, c] = ""

    for c in [
        "reader_pair_found_flag",
        "pair_punch_count",
        "pair_start_min",
        "pair_end_min",
        "pair_span_min",
        "pair_outside_punch_count",
        "pair_next_day_flag",
    ]:
        if c in out.columns:
            out.loc[absent_mask, c] = 0

    out["final_display_value"] = np.where(
        out["final_status_label"] == "ABSENT",
        "ABSENT",
        np.where(
            out["final_status_label"] == "WO_REVIEW",
            "WO_REVIEW",
            np.where(
                out["shift_status"] == "withheld",
                "WITHHELD",
                np.where(
                    out["shift_status"] == "review",
                    out["final_shift"].astype(str) + " (REVIEW)",
                    out["final_shift"].astype(str)
                )
            )
        )
    )

    debug_cols = [
        "PersonCode",
        "PersonCode_norm",
        "attendance_day",
        "attendance_weekday",
        "is_sunday",
        "txn_punch_count",

        "final_shift",
        "final_day_status",
        "final_status_label",
        "final_display_value",
        "shift_status",
        "decision_reason",
        "final_message",

        "prod_model_pred_shift",
        "prod_model_confidence",

        "window_punch_count",
        "reader_in_count",
        "reader_out_count",
        "reader_pair_found_flag",

        "best_shift_candidate",
        "pair_confidence_bucket",

        "pair_first_punch",
        "pair_last_punch",
        "pair_punch_count",
        "pair_start_min",
        "pair_end_min",
        "pair_span_min",
        "pair_next_day_flag",
        "reader_pair_method",
        "reader_pair_out_scope",
        "pair_outside_punch_count",
        "pair_outside_debug_count",
        "pair_ignored_after_out_count",

        "raw_transactions",
        "pair_preview",
        "raw_transactions_debug",
        "pair_preview_debug",
    ]

    debug_cols = [c for c in debug_cols if c in out.columns]
    debug_df = out[debug_cols].copy()

    clean_map = {
        "PersonCode": "Employee ID",
        "attendance_day": "Date",
        "final_shift": "Final Shift",
        "final_status_label": "Final Status Label",
        "final_day_status": "Final Day Status",
        "shift_status": "Shift Status",
        "final_message": "Final Message",
        "txn_punch_count": "Same-Day Punch Count",
        "pair_first_punch": "Punch In",
        "pair_last_punch": "Punch Out",
        "reader_pair_found_flag": "Valid Reader Pair Found",
        "raw_transactions": "Raw Transactions",
        "pair_preview": "Candidate Punch Slice",
        "prod_model_pred_shift": "Model Predicted Shift",
        "prod_model_confidence": "Model Confidence",
        "decision_reason": "Decision Reason",
    }

    clean_cols = [c for c in clean_map.keys() if c in debug_df.columns]
    clean_df = debug_df[clean_cols].rename(columns=clean_map).copy()
    clean_df = clean_df.astype(object)

    non_working_mask = clean_df["Final Status Label"].astype(str).isin(["ABSENT", "WO_REVIEW", "WITHHELD"])

    for c in ["Model Predicted Shift", "Model Confidence"]:
        if c in clean_df.columns:
            clean_df.loc[non_working_mask, c] = ""

    for col in clean_df.columns:
        clean_df[col] = clean_df[col].replace({
            pd.NaT: "",
            "NaT": "",
            "nan": "",
            "NaN": "",
            "None": "",
            np.nan: "",
        })

    if "Date" in clean_df.columns:
        clean_df["Date"] = pd.to_datetime(clean_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")

    if "Model Confidence" in clean_df.columns:
        def _fmt_conf(x):
            if x == "" or pd.isna(x):
                return ""
            try:
                return round(float(x) * 100, 2)
            except Exception:
                return ""
        clean_df["Model Confidence"] = clean_df["Model Confidence"].apply(_fmt_conf)

    result_df = clean_df if return_clean else debug_df
    timings["output_dataframe"] = perf_counter() - started

    if output_dir is not None:
        started = perf_counter()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        clean_path = output_dir / "prediction_clean.csv"
        debug_path = output_dir / "prediction_debug.csv"

        clean_df.to_csv(clean_path, index=False)
        debug_df.to_csv(debug_path, index=False)
        timings["output_write"] = perf_counter() - started

    timings["total"] = perf_counter() - total_started
    result_df.attrs["timings"] = timings
    timing_text = " ".join(f"{key}={value:.3f}s" for key, value in timings.items())
    print(f"V3 day-first inference timing: {timing_text}")
    return result_df


def run_inference_debug(input_file_path, output_dir=None):
    return run_inference(input_file_path, output_dir=output_dir, return_clean=False)
