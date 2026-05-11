import json
import logging
from time import perf_counter
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import lightgbm as lgb


logger = logging.getLogger("shift_prediction")
NS_PER_MINUTE = 60 * 1_000_000_000


class ShiftEngine:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)

        with open(self.artifacts_dir / "label_classes.json", "r", encoding="utf-8") as f:
            self.label_classes = json.load(f)

        with open(self.artifacts_dir / "feature_cols.json", "r", encoding="utf-8") as f:
            self.feature_cols = json.load(f)

        with open(self.artifacts_dir / "shift_meta.json", "r", encoding="utf-8") as f:
            self.shift_meta = json.load(f)

        with open(self.artifacts_dir / "config.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.model = lgb.Booster(model_file=str(self.artifacts_dir / "lightgbm_model.txt"))
        self.modeled_shift_classes = sorted(list(self.shift_meta.keys()))
        self.shift_candidate_meta = []
        for shift_name in self.modeled_shift_classes:
            meta = self.shift_meta[shift_name]
            self.shift_candidate_meta.append({
                "shift": shift_name,
                "start_min": meta["start_min"],
                "end_min": meta["end_min"],
                "duration_min": meta["duration_min"],
                "start_lo": meta["start_min"] - self.config["start_early_tol"],
                "start_hi": meta["start_min"] + self.config["start_late_tol"],
                "end_lo": meta["end_min"] - self.config["end_early_tol"],
                "end_hi": meta["end_min"] + self.config["end_late_tol"],
            })

        self.time_buckets = [
            (0,    300,  "cnt_00_05"),
            (300,  480,  "cnt_05_08"),
            (480,  720,  "cnt_08_12"),
            (720,  840,  "cnt_12_14"),
            (840, 1080,  "cnt_14_18"),
            (1080,1260,  "cnt_18_21"),
            (1260,1440,  "cnt_21_24"),
            (1440,1740,  "cnt_next_00_05"),
            (1740,1920,  "cnt_next_05_08"),
            (1920,2160,  "cnt_next_08_12"),
        ]

    def load_transaction_file(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in [".csv", ".txt", ".tsv"]:
            return pd.read_csv(path, sep=None, engine="python")

        if suffix in [".xls", ".xlsx"]:
            try:
                return pd.read_excel(path)
            except Exception:
                return pd.read_csv(path, sep="\t", low_memory=False)

        raise ValueError(f"Unsupported file type: {suffix}")

    def minutes_from_day0(self, ts, day0):
        return float((pd.Timestamp(ts) - pd.Timestamp(day0)).total_seconds() / 60.0)

    def build_gap_stats(self, ts_series):
        ts_series = pd.to_datetime(ts_series).sort_values().reset_index(drop=True)
        if len(ts_series) < 2:
            return 0.0, 0.0, 0
        gaps = np.diff(ts_series.values.astype("datetime64[s]")).astype("timedelta64[m]").astype(int)
        return float(gaps.max()), float(gaps.mean()), int((gaps > 240).sum())

    def bucket_counts(self, mins):
        out = {}
        mins = pd.Series(mins).astype(float)
        for start_min, end_min, name in self.time_buckets:
            out[name] = int(((mins >= start_min) & (mins < end_min)).sum())
        return out

    def pair_score(self, start_min, end_min, duration_min, shift_name, pair_punch_count, outside_punch_count):
        meta = self.shift_meta[shift_name]

        start_diff = abs(start_min - meta["start_min"])
        end_diff = abs(end_min - meta["end_min"])
        duration_diff = abs(duration_min - meta["duration_min"])

        penalty = 0.0
        penalty += outside_punch_count * 10.0

        if pair_punch_count < 2:
            penalty += 300.0
        if duration_min < self.config["min_valid_span"]:
            penalty += 400.0
        if duration_min > self.config["max_valid_span"]:
            penalty += 500.0

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

    def prepare_transaction_df(self, txn: pd.DataFrame) -> pd.DataFrame:
        txn = txn.copy()

        if "TransactionDateTime" not in txn.columns or "EmpCode" not in txn.columns:
            raise ValueError("Transaction file must contain at least: EmpCode, TransactionDateTime")

        txn["TransactionDateTime"] = pd.to_datetime(txn["TransactionDateTime"], errors="coerce")
        txn = txn[txn["TransactionDateTime"].notna()].copy()

        txn["EmpCode_norm"] = txn["EmpCode"].astype(str).str.strip()

        for c in ["ReaderId", "ReasonCode", "TransactionCode"]:
            if c not in txn.columns:
                txn[c] = np.nan

        txn = txn.sort_values(["EmpCode_norm", "TransactionDateTime"]).reset_index(drop=True)
        return txn

    def build_txn_index(self, txn: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        txn_index = {}
        for emp, g in txn.groupby("EmpCode_norm", sort=False):
            ts = g["TransactionDateTime"].to_numpy(dtype="datetime64[ns]")
            txn_index[emp] = {
                "ts": ts,
                "ts_ns": ts.astype("datetime64[ns]").astype(np.int64),
                "ts_text": pd.DatetimeIndex(ts).astype(str).to_numpy(),
                "reader": g["ReaderId"].to_numpy(),
                "reason": g["ReasonCode"].to_numpy(),
                "txn_code": g["TransactionCode"].to_numpy(),
            }
        return txn_index

    def get_window_df(self, txn_index, emp: str, day0):
        if pd.isna(day0) or emp not in txn_index:
            return pd.DataFrame(columns=["ts", "reader", "reason", "txn_code"])

        arr = txn_index[emp]["ts"]
        readers = txn_index[emp]["reader"]
        reasons = txn_index[emp]["reason"]
        txcodes = txn_index[emp]["txn_code"]

        start = np.datetime64(pd.Timestamp(day0))
        end = np.datetime64(pd.Timestamp(day0) + pd.Timedelta(hours=self.config["window_hours_after"]))

        left = np.searchsorted(arr, start, side="left")
        right = np.searchsorted(arr, end, side="left")

        if right <= left:
            return pd.DataFrame(columns=["ts", "reader", "reason", "txn_code"])

        return pd.DataFrame({
            "ts": pd.to_datetime(arr[left:right]),
            "reader": readers[left:right],
            "reason": reasons[left:right],
            "txn_code": txcodes[left:right],
        }).sort_values("ts").reset_index(drop=True)

    def choose_best_pair_candidate(self, win_df, day0):
        if win_df.empty:
            return None, None, []

        ts = pd.to_datetime(win_df["ts"]).sort_values().reset_index(drop=True)
        mins = ts.apply(lambda x: self.minutes_from_day0(x, day0)).astype(float).tolist()
        n = len(ts)
        candidates = []

        if n == 1:
            m = mins[0]
            for sh in self.modeled_shift_classes:
                meta = self.shift_meta[sh]
                start_diff = abs(m - meta["start_min"])
                end_diff = abs(m - meta["end_min"])
                score = min(start_diff, end_diff) + 500.0
                candidates.append({
                    "shift": sh,
                    "score": float(score),
                    "start_diff": float(start_diff),
                    "end_diff": float(end_diff),
                    "duration_diff": np.nan,
                    "i": 0,
                    "j": 0,
                    "pair_punch_count": 1,
                    "outside_punch_count": 0,
                })

        for sh in self.modeled_shift_classes:
            meta = self.shift_meta[sh]
            start_lo = meta["start_min"] - self.config["start_early_tol"]
            start_hi = meta["start_min"] + self.config["start_late_tol"]
            end_lo = meta["end_min"] - self.config["end_early_tol"]
            end_hi = meta["end_min"] + self.config["end_late_tol"]

            for i in range(n):
                sm = mins[i]
                if sm < start_lo or sm > start_hi:
                    continue

                for j in range(i + 1, n):
                    em = mins[j]
                    if em < end_lo or em > end_hi:
                        continue

                    span = em - sm
                    if span < self.config["min_valid_span"] or span > self.config["max_valid_span"]:
                        continue

                    pair_punch_count = j - i + 1
                    outside_punch_count = n - pair_punch_count

                    scored = self.pair_score(sm, em, span, sh, pair_punch_count, outside_punch_count)

                    candidates.append({
                        "shift": sh,
                        "score": scored["score"],
                        "start_diff": scored["start_diff"],
                        "end_diff": scored["end_diff"],
                        "duration_diff": scored["duration_diff"],
                        "i": i,
                        "j": j,
                        "pair_punch_count": pair_punch_count,
                        "outside_punch_count": outside_punch_count,
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
                -x["pair_punch_count"],
            )
        )

        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        return best, second, ranked

    def extract_pair_features(self, win_df, day0, best, second):
        base = {
            "window_punch_count": int(len(win_df)),
            "window_large_gap_count": 0,
            "window_cluster_count": 0,

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
            "valid_pair_found": False,
            "pair_outside_punch_count": 0,
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
            "pair_manual_continuous_flag": 0,
            "pair_confidence_bucket": "unknown",
        }

        for _, _, name in self.time_buckets:
            base[name] = 0

        if win_df.empty:
            return base

        full_ts = pd.to_datetime(win_df["ts"]).sort_values().reset_index(drop=True)
        _, _, large_gap_count = self.build_gap_stats(full_ts)

        base["window_large_gap_count"] = int(large_gap_count)
        base["window_cluster_count"] = int(large_gap_count + 1) if len(full_ts) > 0 else 0
        base["pair_no_punch_flag"] = 0
        base["raw_transactions"] = " | ".join(full_ts.astype(str).tolist())

        if best is None:
            if len(full_ts) == 1:
                base["pair_single_punch_flag"] = 1
            return base

        base["best_shift_candidate"] = best["shift"]
        base["best_shift_score"] = float(best["score"])
        base["best_shift_start_diff"] = float(best["start_diff"])
        base["best_shift_end_diff"] = float(best["end_diff"])
        base["best_shift_duration_diff"] = float(best["duration_diff"]) if pd.notna(best["duration_diff"]) else np.nan

        if second is not None:
            base["second_shift_candidate"] = second["shift"]
            base["second_shift_score"] = float(second["score"])
            base["candidate_score_gap"] = float(second["score"] - best["score"])

        pair_df = win_df.iloc[best["i"]:best["j"] + 1].copy().reset_index(drop=True)
        pair_ts = pd.to_datetime(pair_df["ts"]).sort_values().reset_index(drop=True)
        pair_mins = pair_ts.apply(lambda x: self.minutes_from_day0(x, day0)).astype(float)

        base["pair_first_punch"] = pd.Timestamp(pair_ts.iloc[0])
        base["pair_last_punch"] = pd.Timestamp(pair_ts.iloc[-1])
        base["pair_start_min"] = float(pair_mins.iloc[0])
        base["pair_end_min"] = float(pair_mins.iloc[-1])
        base["pair_span_min"] = float(pair_mins.iloc[-1] - pair_mins.iloc[0])
        base["pair_punch_count"] = int(len(pair_df))
        base["pair_outside_punch_count"] = int(len(win_df) - len(pair_df))
        base["pair_next_day_flag"] = int(pair_ts.iloc[-1].normalize() > pd.Timestamp(day0).normalize())
        base["pair_preview"] = " | ".join(pair_ts.astype(str).tolist()[:12])
        base["valid_pair_found"] = bool(base["pair_punch_count"] >= 2 and pd.notna(best.get("shift")))

        max_gap, mean_gap, _ = self.build_gap_stats(pair_ts)
        base["pair_max_gap_min"] = float(max_gap)
        base["pair_mean_gap_min"] = float(mean_gap)
        base["pair_unique_reader_count"] = int(pd.Series(pair_df["reader"]).nunique(dropna=True))
        base["pair_unique_reason_count"] = int(pd.Series(pair_df["reason"]).nunique(dropna=True))
        base["pair_unique_txncode_count"] = int(pd.Series(pair_df["txn_code"]).nunique(dropna=True))

        if len(pair_df) == 1:
            base["pair_single_punch_flag"] = 1

        for k, v in self.bucket_counts(pair_mins).items():
            base[k] = v

        chosen_shift = best["shift"]
        expected = self.shift_meta[chosen_shift]["duration_min"]
        span = base["pair_span_min"]

        base["pair_overtime_like_flag"] = int(
            span >= expected + 120 and
            span <= expected + 360 and
            base["best_shift_start_diff"] <= 120
        )

        base["pair_manual_continuous_flag"] = int(
            span > expected + 360 or
            base["pair_outside_punch_count"] >= 8 or
            base["window_cluster_count"] >= 4
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

    def build_feature_base(self, txn_df: pd.DataFrame) -> pd.DataFrame:
        timings = {}
        started = perf_counter()
        txn_df = self.prepare_transaction_df(txn_df)
        timings["transaction_preprocess"] = perf_counter() - started

        started = perf_counter()
        txn_index = self.build_txn_index(txn_df)
        timings["employee_index"] = perf_counter() - started

        started = perf_counter()
        employee_days = (
            txn_df[["EmpCode_norm", "TransactionDateTime"]]
            .assign(attendance_day=lambda x: x["TransactionDateTime"].dt.normalize())
            [["EmpCode_norm", "attendance_day"]]
            .drop_duplicates()
            .sort_values(["EmpCode_norm", "attendance_day"])
            .reset_index(drop=True)
        )
        timings["employee_day_grouping"] = perf_counter() - started

        started = perf_counter()
        rows = []
        for emp, day0 in employee_days.itertuples(index=False, name=None):
            emp = str(emp).strip()
            day0 = pd.Timestamp(day0)

            feat = self.extract_pair_features_fast(txn_index, emp, day0)

            feat["EmpCode_norm"] = emp
            feat["attendance_day"] = day0
            feat["weekday_num"] = day0.weekday()
            feat["is_sunday"] = int(day0.weekday() == 6)
            rows.append(feat)

        feature_df = pd.DataFrame(rows)
        timings["feature_row_generation"] = perf_counter() - started
        logger.info(
            "Shift engine feature build timings: preprocess=%.3fs index=%.3fs employee_day_grouping=%.3fs row_generation=%.3fs rows=%s",
            timings["transaction_preprocess"],
            timings["employee_index"],
            timings["employee_day_grouping"],
            timings["feature_row_generation"],
            len(feature_df),
        )
        return feature_df

    def extract_pair_features_fast(self, txn_index, emp: str, day0: pd.Timestamp) -> dict:
        if emp not in txn_index:
            return self._empty_pair_feature_base(0)

        emp_index = txn_index[emp]
        arr = emp_index["ts"]
        ts_ns = emp_index["ts_ns"]

        day_start = np.datetime64(day0, "ns")
        day_end = np.datetime64(day0 + pd.Timedelta(hours=self.config["window_hours_after"]), "ns")
        left = np.searchsorted(arr, day_start, side="left")
        right = np.searchsorted(arr, day_end, side="left")

        if right <= left:
            return self._empty_pair_feature_base(0)

        window_ts_ns = ts_ns[left:right]
        window_text = emp_index["ts_text"][left:right]
        window_reader = emp_index["reader"][left:right]
        window_reason = emp_index["reason"][left:right]
        window_txn_code = emp_index["txn_code"][left:right]
        day0_ns = day_start.astype(np.int64)
        mins = (window_ts_ns - day0_ns).astype(float) / NS_PER_MINUTE

        best, second = self.choose_best_pair_candidate_fast(mins)
        return self.extract_pair_features_from_arrays(
            mins=mins,
            ts_ns=window_ts_ns,
            ts_text=window_text,
            readers=window_reader,
            reasons=window_reason,
            txn_codes=window_txn_code,
            day0=day0,
            best=best,
            second=second,
        )

    def _empty_pair_feature_base(self, window_punch_count: int) -> dict:
        base = {
            "window_punch_count": int(window_punch_count),
            "window_large_gap_count": 0,
            "window_cluster_count": 0,

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
            "valid_pair_found": False,
            "pair_outside_punch_count": 0,
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
            "pair_manual_continuous_flag": 0,
            "pair_confidence_bucket": "unknown",
        }

        for _, _, name in self.time_buckets:
            base[name] = 0

        return base

    def choose_best_pair_candidate_fast(self, mins: np.ndarray) -> tuple[dict | None, dict | None]:
        n = len(mins)
        if n == 0:
            return None, None

        best = None
        second = None
        best_key = None
        second_key = None
        order = 0

        def push_candidate(candidate):
            nonlocal best, second, best_key, second_key, order
            key = (
                candidate["score"],
                candidate["duration_diff"] if pd.notna(candidate["duration_diff"]) else 999999,
                candidate["start_diff"],
                candidate["end_diff"],
                -candidate["pair_punch_count"],
                order,
            )
            order += 1
            if best_key is None or key < best_key:
                second, second_key = best, best_key
                best, best_key = candidate, key
            elif second_key is None or key < second_key:
                second, second_key = candidate, key

        if n == 1:
            m = float(mins[0])
            for meta in self.shift_candidate_meta:
                sh = meta["shift"]
                start_diff = abs(m - meta["start_min"])
                end_diff = abs(m - meta["end_min"])
                push_candidate({
                    "shift": sh,
                    "score": float(min(start_diff, end_diff) + 500.0),
                    "start_diff": float(start_diff),
                    "end_diff": float(end_diff),
                    "duration_diff": np.nan,
                    "i": 0,
                    "j": 0,
                    "pair_punch_count": 1,
                    "outside_punch_count": 0,
            })
            return best, second

        min_valid_span = self.config["min_valid_span"]
        max_valid_span = self.config["max_valid_span"]
        for meta in self.shift_candidate_meta:
            sh = meta["shift"]
            start_lo = meta["start_lo"]
            start_hi = meta["start_hi"]
            end_lo = meta["end_lo"]
            end_hi = meta["end_hi"]
            shift_start_min = meta["start_min"]
            shift_end_min = meta["end_min"]
            shift_duration_min = meta["duration_min"]

            start_indices = np.flatnonzero((mins >= start_lo) & (mins <= start_hi))
            for i in start_indices:
                j_start = np.searchsorted(mins, end_lo, side="left")
                j_end = np.searchsorted(mins, end_hi, side="right")
                if j_start <= i:
                    j_start = i + 1
                if j_end <= j_start:
                    continue

                for j in range(j_start, j_end):
                    span = float(mins[j] - mins[i])
                    if span < min_valid_span or span > max_valid_span:
                        continue

                    pair_punch_count = int(j - i + 1)
                    outside_punch_count = int(n - pair_punch_count)
                    start_diff = abs(float(mins[i]) - shift_start_min)
                    end_diff = abs(float(mins[j]) - shift_end_min)
                    duration_diff = abs(span - shift_duration_min)
                    score = self.pair_score_value(
                        start_diff,
                        end_diff,
                        duration_diff,
                        span,
                        pair_punch_count,
                        outside_punch_count,
                    )

                    push_candidate({
                        "shift": sh,
                        "score": score,
                        "start_diff": float(start_diff),
                        "end_diff": float(end_diff),
                        "duration_diff": float(duration_diff),
                        "i": int(i),
                        "j": int(j),
                        "pair_punch_count": pair_punch_count,
                        "outside_punch_count": outside_punch_count,
                    })

        return best, second

    def pair_score_value(
        self,
        start_diff: float,
        end_diff: float,
        duration_diff: float,
        duration_min: float,
        pair_punch_count: int,
        outside_punch_count: int,
    ) -> float:
        penalty = outside_punch_count * 10.0

        if pair_punch_count < 2:
            penalty += 300.0
        if duration_min < self.config["min_valid_span"]:
            penalty += 400.0
        if duration_min > self.config["max_valid_span"]:
            penalty += 500.0

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

        return float(
            start_diff * 0.90 +
            end_diff * 0.75 +
            duration_diff * 0.65 +
            penalty -
            reward
        )

    def extract_pair_features_from_arrays(
        self,
        mins: np.ndarray,
        ts_ns: np.ndarray,
        ts_text: np.ndarray,
        readers: np.ndarray,
        reasons: np.ndarray,
        txn_codes: np.ndarray,
        day0: pd.Timestamp,
        best: dict | None,
        second: dict | None,
    ) -> dict:
        base = self._empty_pair_feature_base(len(ts_ns))
        if len(ts_ns) == 0:
            return base

        _, _, large_gap_count = self.gap_stats_from_ns(ts_ns)
        base["window_large_gap_count"] = int(large_gap_count)
        base["window_cluster_count"] = int(large_gap_count + 1)
        base["pair_no_punch_flag"] = 0
        base["raw_transactions"] = " | ".join(ts_text.tolist())

        if best is None:
            if len(ts_ns) == 1:
                base["pair_single_punch_flag"] = 1
            return base

        base["best_shift_candidate"] = best["shift"]
        base["best_shift_score"] = float(best["score"])
        base["best_shift_start_diff"] = float(best["start_diff"])
        base["best_shift_end_diff"] = float(best["end_diff"])
        base["best_shift_duration_diff"] = float(best["duration_diff"]) if pd.notna(best["duration_diff"]) else np.nan

        if second is not None:
            base["second_shift_candidate"] = second["shift"]
            base["second_shift_score"] = float(second["score"])
            base["candidate_score_gap"] = float(second["score"] - best["score"])

        i = int(best["i"])
        j = int(best["j"])
        pair_slice = slice(i, j + 1)
        pair_ts_ns = ts_ns[pair_slice]
        pair_mins = mins[pair_slice]

        base["pair_first_punch"] = pd.Timestamp(pair_ts_ns[0])
        base["pair_last_punch"] = pd.Timestamp(pair_ts_ns[-1])
        base["pair_start_min"] = float(pair_mins[0])
        base["pair_end_min"] = float(pair_mins[-1])
        base["pair_span_min"] = float(pair_mins[-1] - pair_mins[0])
        base["pair_punch_count"] = int(len(pair_ts_ns))
        base["pair_outside_punch_count"] = int(len(ts_ns) - len(pair_ts_ns))
        base["pair_next_day_flag"] = int(pair_ts_ns[-1] >= (np.datetime64(day0, "ns").astype(np.int64) + 24 * 60 * NS_PER_MINUTE))
        base["pair_preview"] = " | ".join(ts_text[pair_slice].tolist()[:12])
        base["valid_pair_found"] = bool(base["pair_punch_count"] >= 2 and pd.notna(best.get("shift")))

        max_gap, mean_gap, _ = self.gap_stats_from_ns(pair_ts_ns)
        base["pair_max_gap_min"] = float(max_gap)
        base["pair_mean_gap_min"] = float(mean_gap)
        base["pair_unique_reader_count"] = self.count_unique_non_null(readers[pair_slice])
        base["pair_unique_reason_count"] = self.count_unique_non_null(reasons[pair_slice])
        base["pair_unique_txncode_count"] = self.count_unique_non_null(txn_codes[pair_slice])

        if len(pair_ts_ns) == 1:
            base["pair_single_punch_flag"] = 1

        for k, v in self.bucket_counts_fast(pair_mins).items():
            base[k] = v

        chosen_shift = best["shift"]
        expected = self.shift_meta[chosen_shift]["duration_min"]
        span = base["pair_span_min"]

        base["pair_overtime_like_flag"] = int(
            span >= expected + 120 and
            span <= expected + 360 and
            base["best_shift_start_diff"] <= 120
        )

        base["pair_manual_continuous_flag"] = int(
            span > expected + 360 or
            base["pair_outside_punch_count"] >= 8 or
            base["window_cluster_count"] >= 4
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

    def gap_stats_from_ns(self, ts_ns: np.ndarray) -> tuple[float, float, int]:
        if len(ts_ns) < 2:
            return 0.0, 0.0, 0
        gaps = np.diff(ts_ns) // NS_PER_MINUTE
        return float(gaps.max()), float(gaps.mean()), int((gaps > 240).sum())

    def bucket_counts_fast(self, mins: np.ndarray) -> dict:
        return {
            name: int(((mins >= start_min) & (mins < end_min)).sum())
            for start_min, end_min, name in self.time_buckets
        }

    def count_unique_non_null(self, values: np.ndarray) -> int:
        if len(values) == 0:
            return 0
        return int(pd.unique(pd.Series(values).dropna()).size)

    def score_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        started = perf_counter()
        df = feature_df.copy()

        X = df[self.feature_cols].copy()
        for c in self.feature_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(-999)

        prob = self.model.predict(X[self.feature_cols])
        prob = np.array(prob)
        if prob.ndim == 1:
            prob = prob.reshape(-1, 1)

        best_idx = prob.argmax(axis=1)
        df["prod_model_pred_shift"] = [self.label_classes[i] for i in best_idx]
        df["prod_model_confidence"] = prob.max(axis=1)

        for i, cls in enumerate(self.label_classes):
            df[f"prod_prob_{cls}"] = prob[:, i]

        logger.info("Shift engine model inference timing: rows=%s seconds=%.3f", len(df), perf_counter() - started)
        return df

    def apply_business_rules(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        started = perf_counter()
        df = scored_df.copy()

        final_shift = []
        final_day_status = []
        shift_status = []
        final_status_label = []
        final_message = []
        working_shift_hint = []

        for _, row in df.iterrows():
            model_pred = row.get("prod_model_pred_shift", np.nan)
            conf = float(row.get("prod_model_confidence", 0) or 0)
            pair_pred = row.get("best_shift_candidate", np.nan)
            pair_bucket = str(row.get("pair_confidence_bucket", "unknown"))

            is_sunday = int(row.get("is_sunday", 0) or 0)
            pair_single = int(row.get("pair_single_punch_flag", 0) or 0)
            pair_manual = int(row.get("pair_manual_continuous_flag", 0) or 0)
            valid_pair_found = row.get("valid_pair_found", False)
            if pd.isna(valid_pair_found):
                valid_pair_found = False
            elif isinstance(valid_pair_found, str):
                valid_pair_found = valid_pair_found.strip().lower() in {"true", "1", "yes", "y"}
            else:
                valid_pair_found = bool(valid_pair_found)
            no_valid_pair = int(not valid_pair_found)
            overtime_like = int(row.get("pair_overtime_like_flag", 0) or 0)

            start_diff = pd.to_numeric(row.get("best_shift_start_diff", np.nan), errors="coerce")
            dur_diff = pd.to_numeric(row.get("best_shift_duration_diff", np.nan), errors="coerce")
            next_day = int(row.get("pair_next_day_flag", 0) or 0)

            if is_sunday == 1:
                final_shift.append("WO")
                final_day_status.append("WO")
                shift_status.append("review")
                final_status_label.append("WO_REVIEW")
                final_message.append("Worked on weekly off / possible overtime or extra duty")
                working_shift_hint.append(model_pred)
                continue

            if pair_single == 1:
                final_shift.append("unknown")
                final_day_status.append("WITHHELD")
                shift_status.append("withheld")
                final_status_label.append("WITHHELD")
                final_message.append("Only one transaction found")
                working_shift_hint.append(model_pred)
                continue

            if no_valid_pair == 1:
                final_shift.append("unknown")
                final_day_status.append("WITHHELD")
                shift_status.append("withheld")
                final_status_label.append("WITHHELD")
                final_message.append("No valid in-out pair found")
                working_shift_hint.append(model_pred)
                continue

            if pair_manual == 1:
                final_shift.append(model_pred)
                final_day_status.append("WORKING")
                shift_status.append("review")
                final_status_label.append("SHIFT_REVIEW")
                final_message.append("Continuous/manual-like pattern; review required")
                working_shift_hint.append(model_pred)
                continue

            if conf >= 0.995:
                final_shift.append(model_pred)
                final_day_status.append("WORKING")
                shift_status.append("predicted")
                final_status_label.append("SHIFT_PREDICTED")
                final_message.append("High-confidence shift prediction")
                working_shift_hint.append(model_pred)
                continue

            if conf >= 0.98 and model_pred == pair_pred:
                final_shift.append(model_pred)
                final_day_status.append("WORKING")
                shift_status.append("predicted")
                final_status_label.append("SHIFT_PREDICTED")
                final_message.append("Model and pair strongly agree")
                working_shift_hint.append(model_pred)
                continue

            if (
                model_pred == pair_pred and
                next_day == 1 and
                pd.notna(start_diff) and start_diff <= 120 and
                pd.notna(dur_diff) and dur_diff <= 120 and
                conf >= 0.92
            ):
                final_shift.append(model_pred)
                final_day_status.append("WORKING")
                shift_status.append("predicted")
                final_status_label.append("SHIFT_PREDICTED")
                final_message.append("Overnight shift anchored by pair pattern")
                working_shift_hint.append(model_pred)
                continue

            if conf >= 0.95 and (pd.isna(pair_pred) or pair_pred == model_pred or pair_bucket in {"medium", "unknown"}):
                final_shift.append(model_pred)
                final_day_status.append("WORKING")
                shift_status.append("predicted")
                final_status_label.append("SHIFT_PREDICTED")
                final_message.append("Good-confidence shift prediction")
                working_shift_hint.append(model_pred)
                continue

            if conf >= 0.85 and model_pred == pair_pred:
                final_shift.append(model_pred)
                final_day_status.append("WORKING")
                shift_status.append("review")
                final_status_label.append("SHIFT_REVIEW")
                final_message.append("Model and pair agree; review suggested")
                working_shift_hint.append(model_pred)
                continue

            if overtime_like == 1 and conf >= 0.80:
                final_shift.append(model_pred)
                final_day_status.append("WORKING")
                shift_status.append("review")
                final_status_label.append("SHIFT_REVIEW")
                final_message.append("Overtime-like pattern; review suggested")
                working_shift_hint.append(model_pred)
                continue

            if conf >= 0.75:
                final_shift.append(model_pred)
                final_day_status.append("WORKING")
                shift_status.append("review")
                final_status_label.append("SHIFT_REVIEW")
                final_message.append("Low-confidence model output; review suggested")
                working_shift_hint.append(model_pred)
                continue

            final_shift.append("unknown")
            final_day_status.append("WITHHELD")
            shift_status.append("withheld")
            final_status_label.append("WITHHELD")
            final_message.append("Ambiguous row")
            working_shift_hint.append(model_pred)

        df["working_shift_hint"] = working_shift_hint
        df["final_shift"] = final_shift
        df["final_day_status"] = final_day_status
        df["shift_status"] = shift_status
        df["final_status_label"] = final_status_label
        df["final_message"] = final_message

        logger.info("Shift engine final business rules timing: rows=%s seconds=%.3f", len(df), perf_counter() - started)
        return df

    def predict_from_transaction_df(self, txn_df: pd.DataFrame) -> pd.DataFrame:
        total_started = perf_counter()
        feature_df = self.build_feature_base(txn_df)
        scored_df = self.score_features(feature_df)
        final_df = self.apply_business_rules(scored_df)

        keep_cols = [
            "EmpCode_norm", "attendance_day", "weekday_num", "is_sunday",
            "window_punch_count", "pair_punch_count", "valid_pair_found",
            "best_shift_candidate", "pair_confidence_bucket",
            "prod_model_pred_shift", "prod_model_confidence", "working_shift_hint",
            "final_day_status", "final_shift", "shift_status", "final_status_label", "final_message",
            "pair_start_min", "pair_end_min", "pair_span_min", "pair_next_day_flag", "pair_preview", "raw_transactions",
        ] + [c for c in final_df.columns if c.startswith("prod_prob_")]

        keep_cols = [c for c in keep_cols if c in final_df.columns]
        output = final_df[keep_cols].copy()
        logger.info("Shift engine total inference timing: rows=%s seconds=%.3f", len(output), perf_counter() - total_started)
        return output

    def predict_from_file(self, file_path: str) -> pd.DataFrame:
        txn_df = self.load_transaction_file(file_path)
        return self.predict_from_transaction_df(txn_df)
