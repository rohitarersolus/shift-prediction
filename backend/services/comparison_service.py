from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Optional, Union

import pandas as pd
import numpy as np

from backend.config import settings
from backend.services.file_service import AppError


EMPLOYEE_ALIASES = (
    "EmpCode_norm",
    "EmpCode",
    "PersonCode",
    "EmployeeCode",
    "EmployeeId",
    "EmployeeID",
)
PREDICTION_DATE_ALIASES = ("attendance_day", "AttendanceDate", "AccountingDate", "Date")
ATTENDANCE_DATE_ALIASES = ("AttendanceDate", "AccountingDate", "attendance_day", "Date")
PREDICTED_SHIFT_ALIASES = ("final_shift", "Final Shift", "Predicted Shift")
PREDICTED_STATUS_ALIASES = ("final_status_label", "Final Status Label", "Prediction Status", "shift_status", "final_day_status")
PREDICTED_DAY_STATUS_ALIASES = ("final_day_status", "Final Day Status")
ATTENDANCE_SHIFT_ALIASES = (
    "ShiftShortName",
    "AssignedShiftShortName",
    "Shift",
    "FinalShift",
    "final_shift",
)
ATTENDANCE_STATUS_ALIASES = (
    "AttendanceStatus",
    "AbsentStatus",
    "DayStatus",
    "Status",
    "FHStatus",
    "SHStatus",
)
PUNCH_IN_ALIASES = (
    "punch_in_time",
    "Punch In",
    "PunchIn",
    "PunchInTime",
    "In Punch",
    "InPunch",
    "InPunchTime",
    "In Time",
    "InTime",
    "First In",
    "FirstIn",
    "FirstInTime",
    "First Punch",
    "FirstPunch",
    "FirstPunchTime",
    "Actual In",
    "ActualIn",
    "ActualInTime",
)
PUNCH_OUT_ALIASES = (
    "punch_out_time",
    "Punch Out",
    "PunchOut",
    "PunchOutTime",
    "Out Punch",
    "OutPunch",
    "OutPunchTime",
    "Out Time",
    "OutTime",
    "Last Out",
    "LastOut",
    "LastOutTime",
    "Last Punch",
    "LastPunch",
    "LastPunchTime",
    "Actual Out",
    "ActualOut",
    "ActualOutTime",
)
RAW_TRANSACTION_DATE_ALIASES = ("TransactionDateTime", "TransactionDate", "TransactionTime")
PREDICTION_PUNCH_SOURCE_ALIASES = ("raw_transactions", "Raw Transactions", "candidate_punch_slice", "Candidate Punch Slice", "pair_preview")
NULL_STRINGS = {"", "NULL", "NAN", "NONE", "NAT"}
BASE_SHIFT_CODES = {"KFS", "KSS", "PF", "PGM", "PS", "PT"}
SUPPORTED_WORKING_SHIFT_CODES = {"PF", "PGM", "PS", "PT"}
NON_WORKING_SHIFT_CODES = {"WO", "OFF", "HOLIDAY", "LEAVE", "ABSENT"}
logger = logging.getLogger("shift_prediction")
MAX_COMPARISON_DF_CACHE_SIZE = 3
COMPARISON_SEARCH_COLUMNS = (
    "employee_id",
    "date",
    "attendance_punch_in_time",
    "attendance_punch_out_time",
    "transaction_first_in_time",
    "transaction_last_out_time",
    "attendance_truth_shift_family",
    "predicted_final_shift",
    "predicted_final_status_label",
    "predicted_final_day_status",
    "attendance_truth_shift",
    "attendance_truth_status",
    "comparison_layer",
    "comparison_result",
    "mismatch_reason",
)
ATTENDANCE_SHIFT_FAMILY_MAP = {
    "PFW": "PF",
    "PF": "PF",
    "PSW": "PS",
    "PS": "PS",
    "PTW": "PT",
    "PT": "PT",
    "PG": "PGM",
    "PGM": "PGM",
    "PGW": "PGM",
}


@dataclass
class ComparisonResult:
    output_file_name: str
    output_path: Path
    clean_output_file_name: str
    clean_output_path: Path
    generated_at: datetime
    summary: dict[str, Union[float, int]]
    columns: list[str]
    records: list[dict[str, Any]]


@dataclass
class ComparisonPageResult:
    output_file_name: str
    columns: list[str]
    page: int
    page_size: int
    total_pages: int
    total_rows: int
    filtered_row_count: int
    available_results: list[str]
    records: list[dict[str, Any]]


class ComparisonService:
    def __init__(self) -> None:
        self._comparison_df_cache: dict[str, pd.DataFrame] = {}

    def compare(
        self,
        attendance_path: Path,
        attendance_name: str,
        prediction_path: Optional[Path] = None,
        prediction_name: Optional[str] = None,
        include_records: bool = True,
    ) -> ComparisonResult:
        total_started = perf_counter()
        timings: dict[str, float] = {}

        if prediction_path is None:
            prediction_path = self._resolve_prediction_output(prediction_name)
            prediction_name = prediction_path.name

        started = perf_counter()
        prediction_df = self._read_table(prediction_path)
        timings["prediction_file_load"] = perf_counter() - started

        started = perf_counter()
        attendance_df = self._read_table(attendance_path)
        timings["attendance_file_load"] = perf_counter() - started

        started = perf_counter()
        prediction_rows = self._normalize_prediction_rows(prediction_df)
        timings["prediction_normalization"] = perf_counter() - started

        started = perf_counter()
        attendance_rows = self._normalize_attendance_rows(attendance_df)
        timings["attendance_normalization"] = perf_counter() - started

        started = perf_counter()
        comparison_df = self._build_comparison(prediction_rows, attendance_rows)
        timings["merge_and_label"] = perf_counter() - started
        self._warn_if_matched_working_predictions_missing_shift(comparison_df)

        if comparison_df.empty:
            raise AppError("No comparable rows were found in the prediction and attendance files.")

        started = perf_counter()
        summary = self._build_summary(comparison_df)
        timings["summary"] = perf_counter() - started

        started = perf_counter()
        output_path = self._write_output_csv(comparison_df, prediction_name or "prediction", attendance_name)
        clean_output_path = self._write_clean_output_csv(comparison_df, prediction_name or "prediction", attendance_name)
        timings["csv_generation"] = perf_counter() - started

        self._cache_comparison_df(output_path.name, comparison_df)

        started = perf_counter()
        records = self._to_records(comparison_df) if include_records else []
        timings["response_serialization"] = perf_counter() - started
        timings["total"] = perf_counter() - total_started

        logger.info(
            "Comparison timing summary: prediction_load=%.3fs attendance_load=%.3fs prediction_normalize=%.3fs attendance_normalize=%.3fs merge_label=%.3fs summary=%.3fs csv=%.3fs response=%.3fs total=%.3fs rows=%s",
            timings["prediction_file_load"],
            timings["attendance_file_load"],
            timings["prediction_normalization"],
            timings["attendance_normalization"],
            timings["merge_and_label"],
            timings["summary"],
            timings["csv_generation"],
            timings["response_serialization"],
            timings["total"],
            len(comparison_df),
        )

        return ComparisonResult(
            output_file_name=output_path.name,
            output_path=output_path,
            clean_output_file_name=clean_output_path.name,
            clean_output_path=clean_output_path,
            generated_at=datetime.now(),
            summary=summary,
            columns=comparison_df.columns.tolist(),
            records=records,
        )

    def get_comparison_page(
        self,
        comparison_name: str,
        *,
        page: int = 1,
        page_size: int = 25,
        search: str = "",
        result: str = "ALL",
    ) -> ComparisonPageResult:
        comparison_df = self._load_comparison_df(comparison_name)
        filtered_df = self._filter_comparison_df(
            comparison_df,
            search=search,
            result=result,
        )

        safe_page_size = max(1, min(page_size, 200))
        filtered_row_count = int(len(filtered_df))
        total_rows = int(len(comparison_df))
        total_pages = max(1, (filtered_row_count + safe_page_size - 1) // safe_page_size)
        safe_page = max(1, min(page, total_pages))
        start_index = (safe_page - 1) * safe_page_size
        page_df = filtered_df.iloc[start_index : start_index + safe_page_size].copy()

        return ComparisonPageResult(
            output_file_name=Path(comparison_name).name,
            columns=comparison_df.columns.tolist(),
            page=safe_page,
            page_size=safe_page_size,
            total_pages=total_pages,
            total_rows=total_rows,
            filtered_row_count=filtered_row_count,
            available_results=self._sorted_unique_values(comparison_df, "comparison_result"),
            records=self._to_records(page_df),
        )

    def _resolve_prediction_output(self, prediction_name: Optional[str]) -> Path:
        outputs_dir = settings.outputs_dir.resolve()

        if prediction_name:
            candidate_path = (settings.outputs_dir / Path(prediction_name).name).resolve()
            if outputs_dir not in candidate_path.parents or not candidate_path.exists():
                raise AppError("Selected prediction output file was not found.", status_code=404)
            return candidate_path

        prediction_outputs = sorted(
            (
                path
                for path in settings.outputs_dir.glob("*-predictions-*.csv")
                if not path.name.startswith("comparison-")
                and "-clean-" not in path.name
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not prediction_outputs:
            raise AppError("No prediction output is available. Run prediction first or upload a prediction CSV.")

        return prediction_outputs[0].resolve()

    def _cache_comparison_df(self, output_file_name: str, comparison_df: pd.DataFrame) -> None:
        cache_key = Path(output_file_name).name
        self._comparison_df_cache[cache_key] = comparison_df.copy()

        while len(self._comparison_df_cache) > MAX_COMPARISON_DF_CACHE_SIZE:
            oldest_key = next(iter(self._comparison_df_cache))
            if oldest_key == cache_key and len(self._comparison_df_cache) == 1:
                break
            self._comparison_df_cache.pop(oldest_key, None)

    def _resolve_comparison_output_path(self, comparison_name: str) -> Path:
        candidate_path = (settings.outputs_dir / Path(comparison_name).name).resolve()
        outputs_dir = settings.outputs_dir.resolve()

        if (
            outputs_dir not in candidate_path.parents
            or not candidate_path.exists()
            or not candidate_path.name.startswith("comparison-")
        ):
            raise AppError("Requested comparison output was not found.", status_code=404)

        return candidate_path

    def _load_comparison_df(self, comparison_name: str) -> pd.DataFrame:
        cache_key = Path(comparison_name).name
        cached_df = self._comparison_df_cache.get(cache_key)
        if cached_df is not None:
            return cached_df.copy()

        output_path = self._resolve_comparison_output_path(cache_key)
        try:
            comparison_df = pd.read_csv(output_path, low_memory=False)
        except Exception as exc:
            logger.exception("Failed to read comparison output: %s", output_path)
            raise AppError("Comparison output could not be loaded.", status_code=500) from exc

        self._cache_comparison_df(cache_key, comparison_df)
        return comparison_df.copy()

    def _filter_comparison_df(
        self,
        comparison_df: pd.DataFrame,
        *,
        search: str = "",
        result: str = "ALL",
    ) -> pd.DataFrame:
        filtered_df = comparison_df

        if result and result != "ALL" and "comparison_result" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["comparison_result"].fillna("").astype(str) == result]

        search_text = search.strip().lower()
        if not search_text:
            return filtered_df

        searchable_columns = [column for column in COMPARISON_SEARCH_COLUMNS if column in filtered_df.columns]
        if not searchable_columns:
            return filtered_df

        mask = pd.Series(False, index=filtered_df.index)
        for column in searchable_columns:
            mask = mask | filtered_df[column].fillna("").astype(str).str.lower().str.contains(search_text, regex=False)

        return filtered_df[mask]

    def _read_table(self, path: Path) -> pd.DataFrame:
        suffix = path.suffix.lower()

        if suffix in {".csv", ".txt", ".tsv"}:
            df = pd.read_csv(path, sep=None, engine="python")
        elif suffix in {".xls", ".xlsx"}:
            try:
                df = pd.read_excel(path)
            except Exception:
                try:
                    df = pd.read_csv(path, sep="\t", engine="python")
                except Exception:
                    df = pd.read_csv(path, sep=None, engine="python")
        else:
            raise AppError(f"Unsupported comparison file type: {suffix}")

        return self._promote_header_if_needed(df)

    def _promote_header_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(how="all").copy()
        df.columns = [self._clean_column_name(column) for column in df.columns]

        if self._has_required_columns(df.columns):
            return df

        for row_index in range(min(10, len(df))):
            values = [self._clean_column_name(value) for value in df.iloc[row_index].tolist()]
            if self._has_required_columns(values):
                promoted = df.iloc[row_index + 1 :].copy()
                promoted.columns = values
                promoted = promoted.dropna(how="all")
                return promoted.reset_index(drop=True)

        return df

    def _has_required_columns(self, columns: Union[list[str], pd.Index]) -> bool:
        return (
            self._find_column(columns, EMPLOYEE_ALIASES) is not None
            and (
                self._find_column(columns, PREDICTION_DATE_ALIASES) is not None
                or self._find_column(columns, ATTENDANCE_DATE_ALIASES) is not None
            )
        )

    def _normalize_prediction_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        emp_col = self._require_column(df, EMPLOYEE_ALIASES, "prediction employee id")
        date_col = self._find_column(df.columns, PREDICTION_DATE_ALIASES)
        shift_col = self._find_column(df.columns, PREDICTED_SHIFT_ALIASES)
        status_col = self._find_column(df.columns, PREDICTED_STATUS_ALIASES)
        final_day_col = self._find_column(df.columns, PREDICTED_DAY_STATUS_ALIASES)
        punch_in_col = self._find_column(df.columns, PUNCH_IN_ALIASES)
        punch_out_col = self._find_column(df.columns, PUNCH_OUT_ALIASES)
        punch_source_col = self._find_column(df.columns, PREDICTION_PUNCH_SOURCE_ALIASES)

        if date_col is None:
            if self._find_column(df.columns, RAW_TRANSACTION_DATE_ALIASES):
                raise AppError(
                    "Prediction override must be the generated prediction CSV, not the raw transaction file. "
                    "Clear the prediction override field to use the latest prediction output from this session."
                )
            raise AppError("Missing required prediction date column.")

        if shift_col is None:
            raise AppError(
                "Prediction override must include generated prediction columns such as `final_shift` and "
                "`final_status_label`. Clear the override field to use the latest prediction output."
            )

        rows = pd.DataFrame({
            "employee_id": df[emp_col].map(self._normalize_employee),
            "date": self._normalize_date_series(df[date_col]),
            "predicted_final_shift": self._clean_series(df[shift_col]),
            "predicted_final_status_label": (
                self._clean_series(df[status_col])
                if status_col
                else pd.Series("", index=df.index)
            ),
            "predicted_final_day_status": (
                self._clean_series(df[final_day_col])
                if final_day_col
                else pd.Series("", index=df.index)
            ),
            "transaction_first_in_time": self._build_punch_series(df, punch_in_col, punch_source_col, "first"),
            "transaction_last_out_time": self._build_punch_series(df, punch_out_col, punch_source_col, "last"),
        })
        rows["predicted_shift_norm"] = rows["predicted_final_shift"].map(self._normalize_shift)
        rows["prediction_status_category"] = self._prediction_status_category_series(rows)
        rows["prediction_shift_bucket"] = self._prediction_shift_bucket_series(rows)
        return self._dedupe_keyed_rows(rows)

    def _normalize_attendance_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        emp_col = self._require_column(df, EMPLOYEE_ALIASES, "attendance employee id")
        date_col = self._require_column(df, ATTENDANCE_DATE_ALIASES, "attendance date")
        shift_col = self._find_column(df.columns, ATTENDANCE_SHIFT_ALIASES)
        status_col = self._find_column(df.columns, ATTENDANCE_STATUS_ALIASES)
        punch_in_col = self._find_column(df.columns, PUNCH_IN_ALIASES)
        punch_out_col = self._find_column(df.columns, PUNCH_OUT_ALIASES)

        if shift_col is None and status_col is None:
            raise AppError("Attendance file is missing shift/status truth columns.")

        truth_shift = (
            self._clean_series(df[shift_col])
            if shift_col
            else pd.Series("", index=df.index)
        )
        truth_status = self._attendance_status_series(df, status_col)

        rows = pd.DataFrame({
            "employee_id": df[emp_col].map(self._normalize_employee),
            "date": self._normalize_date_series(df[date_col]),
            "attendance_truth_shift": truth_shift,
            "attendance_truth_status": truth_status,
            "attendance_punch_in_time": self._build_punch_series(df, punch_in_col, None, "first"),
            "attendance_punch_out_time": self._build_punch_series(df, punch_out_col, None, "last"),
        })
        rows["attendance_truth_shift_family"] = rows["attendance_truth_shift"].map(self._normalize_attendance_shift_family)
        rows["attendance_status_category"] = self._attendance_status_category_series(rows)
        return self._dedupe_keyed_rows(rows)

    def _attendance_status_series(self, df: pd.DataFrame, status_col: Optional[str]) -> pd.Series:
        if status_col and status_col in {"FHStatus", "SHStatus"}:
            fh_col = self._find_column(df.columns, ("FHStatus",))
            sh_col = self._find_column(df.columns, ("SHStatus",))
            fh_values = self._clean_series(df[fh_col]) if fh_col else pd.Series("", index=df.index)
            sh_values = self._clean_series(df[sh_col]) if sh_col else pd.Series("", index=df.index)
            return (fh_values + "/" + sh_values).str.strip("/")

        if status_col:
            return self._clean_series(df[status_col])

        return pd.Series("", index=df.index)

    def _build_comparison(self, prediction_rows: pd.DataFrame, attendance_rows: pd.DataFrame) -> pd.DataFrame:
        merged = prediction_rows.merge(
            attendance_rows,
            on=["employee_id", "date"],
            how="outer",
            indicator=True,
        ).sort_values(["employee_id", "date"], kind="stable").reset_index(drop=True)

        source = merged["_merge"].astype(str)
        pred_shift = merged["predicted_shift_norm"].fillna("").astype(str)
        truth_shift = merged["attendance_truth_shift_family"].fillna("").astype(str)
        truth_category = merged["attendance_status_category"].fillna("").astype(str)
        prediction_category = merged["prediction_status_category"].fillna("").astype(str)
        shift_bucket = merged["prediction_shift_bucket"].fillna("").astype(str)
        predicted_value = self._prediction_compare_value_series(pred_shift, prediction_category)
        truth_value = self._truth_compare_value_series(truth_shift, truth_category)

        left_only = source.eq("left_only")
        right_only = source.eq("right_only")
        matched = source.eq("both")
        supported_working_shift = (
            prediction_category.eq("WORKING")
            & truth_category.eq("WORKING")
            & predicted_value.isin(SUPPORTED_WORKING_SHIFT_CODES)
            & truth_value.isin(SUPPORTED_WORKING_SHIFT_CODES)
        )
        comparable_shift = matched & supported_working_shift
        shift_matches = comparable_shift & predicted_value.eq(truth_value)
        review_bucket = shift_bucket.eq("REVIEW")

        comparison_layer = pd.Series("SHIFT", index=merged.index)
        comparison_result = pd.Series("NON_WORKING_EXCLUDED", index=merged.index)
        mismatch_reason = pd.Series("excluded non-working truth row", index=merged.index)

        comparison_layer.loc[left_only | right_only] = "COVERAGE"
        comparison_result.loc[left_only] = "MISSING_IN_ATTENDANCE"
        mismatch_reason.loc[left_only] = "no matching attendance row"
        comparison_result.loc[right_only] = "MISSING_IN_PREDICTION"
        mismatch_reason.loc[right_only] = "no matching prediction row"

        exact_match = comparable_shift & ~review_bucket & shift_matches
        exact_mismatch = comparable_shift & ~review_bucket & ~shift_matches
        review_match = comparable_shift & review_bucket & shift_matches
        review_mismatch = comparable_shift & review_bucket & ~shift_matches
        status_layer = matched & (
            ~supported_working_shift
            | truth_category.ne("WORKING")
            | prediction_category.ne("WORKING")
        )
        status_match = status_layer & predicted_value.ne("") & truth_value.ne("") & predicted_value.eq(truth_value)
        status_mismatch = status_layer & ~status_match

        comparison_result.loc[exact_match] = "MATCH"
        mismatch_reason.loc[exact_match] = ""
        comparison_result.loc[exact_mismatch] = "SHIFT_MISMATCH"
        mismatch_reason.loc[exact_mismatch] = "prediction does not match attendance answer key"
        comparison_result.loc[review_match] = "REVIEW_MATCH"
        mismatch_reason.loc[review_match] = ""
        comparison_result.loc[review_mismatch] = "REVIEW_MISMATCH"
        mismatch_reason.loc[review_mismatch] = "review prediction does not match attendance answer key"
        comparison_layer.loc[status_layer] = "STATUS"
        comparison_result.loc[status_match] = "STATUS_MATCH"
        mismatch_reason.loc[status_match] = ""
        comparison_result.loc[status_mismatch] = "STATUS_MISMATCH"
        mismatch_reason.loc[status_mismatch] = "prediction status does not match attendance status/shift; excluded from working shift accuracy"

        return pd.DataFrame({
            "employee_id": merged["employee_id"].fillna(""),
            "date": merged["date"].fillna(""),
            "prediction_row_found": np.where(~right_only, "yes", "no"),
            "attendance_punch_in_time": merged["attendance_punch_in_time"].fillna(""),
            "attendance_punch_out_time": merged["attendance_punch_out_time"].fillna(""),
            "transaction_first_in_time": merged["transaction_first_in_time"].fillna(""),
            "transaction_last_out_time": merged["transaction_last_out_time"].fillna(""),
            "predicted_final_shift": merged["predicted_final_shift"].fillna(""),
            "predicted_final_status_label": merged["predicted_final_status_label"].fillna(""),
            "predicted_final_day_status": merged["predicted_final_day_status"].fillna(""),
            "attendance_truth_shift": merged["attendance_truth_shift"].fillna(""),
            "attendance_truth_shift_family": merged["attendance_truth_shift_family"].fillna(""),
            "attendance_truth_status": merged["attendance_truth_status"].fillna(""),
            "attendance_status_bucket": merged["attendance_status_category"].fillna(""),
            "comparison_layer": comparison_layer,
            "comparison_result": comparison_result,
            "mismatch_reason": mismatch_reason,
        })

    def _warn_if_matched_working_predictions_missing_shift(self, comparison_df: pd.DataFrame) -> None:
        if comparison_df.empty:
            return

        working_with_prediction = (
            comparison_df["attendance_status_bucket"].eq("WORKING")
            & comparison_df["prediction_row_found"].eq("yes")
            & comparison_df["predicted_final_shift"].map(self._clean_value).eq("")
        )
        missing_count = int(working_with_prediction.sum())
        if missing_count == 0:
            return

        sample_columns = [
            "employee_id",
            "date",
            "prediction_row_found",
            "predicted_final_status_label",
            "predicted_final_day_status",
            "attendance_truth_shift_family",
            "comparison_result",
        ]
        sample = comparison_df.loc[working_with_prediction, sample_columns].head(10)
        logger.warning(
            "Comparison found %s matched working rows with a prediction row but blank final shift. Sample: %s",
            missing_count,
            sample.to_dict("records"),
        )

    def _compare_matched_row(self, row: pd.Series) -> tuple[str, str, str]:
        pred_shift = self._clean_value(row.get("predicted_shift_norm", ""))
        truth_shift = self._clean_value(row.get("attendance_truth_shift_family", ""))
        truth_category = self._clean_value(row.get("attendance_status_category", ""))
        prediction_category = self._clean_value(row.get("prediction_status_category", ""))
        shift_bucket = self._clean_value(row.get("prediction_shift_bucket", ""))
        pred_value = self._prediction_compare_value(pred_shift, prediction_category)
        truth_value = self._truth_compare_value(truth_shift, truth_category)

        if not pred_value or not truth_value:
            return "SHIFT", "NON_WORKING_EXCLUDED", "excluded non-working truth row"

        shift_matches = pred_value == truth_value

        if shift_bucket == "REVIEW":
            if shift_matches:
                return "SHIFT", "REVIEW_MATCH", ""
            return "SHIFT", "REVIEW_MISMATCH", "review prediction does not match attendance answer key"

        if shift_matches:
            return "SHIFT", "MATCH", ""

        return "SHIFT", "SHIFT_MISMATCH", "prediction does not match attendance answer key"

    def _build_summary(self, comparison_df: pd.DataFrame) -> dict[str, Union[float, int]]:
        attendance_total_rows = int(comparison_df["comparison_result"].ne("MISSING_IN_ATTENDANCE").sum())
        prediction_total_rows = int(comparison_df["comparison_result"].ne("MISSING_IN_PREDICTION").sum())
        missing_in_prediction = int(comparison_df["comparison_result"].eq("MISSING_IN_PREDICTION").sum())
        missing_in_attendance = int(comparison_df["comparison_result"].eq("MISSING_IN_ATTENDANCE").sum())
        matched_rows = int(len(comparison_df) - missing_in_prediction - missing_in_attendance)
        coverage_percent = self._percent(matched_rows, attendance_total_rows)

        comparable_df = comparison_df[
            comparison_df["comparison_layer"].eq("SHIFT")
            & comparison_df["comparison_result"].ne("NON_WORKING_EXCLUDED")
        ].copy()
        comparable_shift_rows = int(len(comparable_df))

        shift_counts = comparable_df["comparison_result"].value_counts()
        exact_shift_matches = int(shift_counts.get("MATCH", 0))
        exact_shift_mismatches = int(shift_counts.get("SHIFT_MISMATCH", 0))
        review_shift_matches = int(shift_counts.get("REVIEW_MATCH", 0))
        review_shift_mismatches = int(shift_counts.get("REVIEW_MISMATCH", 0))
        non_working_excluded = int(
            (
                comparison_df["comparison_result"].eq("NON_WORKING_EXCLUDED")
                | comparison_df["comparison_layer"].eq("STATUS")
            ).sum()
        )
        shift_mismatches = exact_shift_mismatches + review_shift_mismatches

        predicted_only_total = exact_shift_matches + exact_shift_mismatches
        assigned_total = predicted_only_total + review_shift_matches + review_shift_mismatches
        predicted_only_shift_accuracy_percent = self._percent(exact_shift_matches, predicted_only_total)
        assigned_shift_accuracy_percent = self._percent(
            exact_shift_matches + review_shift_matches,
            assigned_total,
        )

        return {
            "attendance_total_rows": attendance_total_rows,
            "prediction_total_rows": prediction_total_rows,
            "matched_rows": matched_rows,
            "missing_in_prediction": missing_in_prediction,
            "missing_in_attendance": missing_in_attendance,
            "coverage_percent": coverage_percent,
            "comparable_working_shift_rows": comparable_shift_rows,
            "comparable_shift_rows": comparable_shift_rows,
            "predicted_only_shift_accuracy_percent": predicted_only_shift_accuracy_percent,
            "assigned_shift_accuracy_percent": assigned_shift_accuracy_percent,
            "exact_shift_matches": exact_shift_matches,
            "exact_shift_mismatches": exact_shift_mismatches,
            "review_shift_matches": review_shift_matches,
            "review_shift_mismatches": review_shift_mismatches,
            "non_working_excluded": non_working_excluded,
            "shift_mismatches": shift_mismatches,
        }

    def _write_output_csv(self, comparison_df: pd.DataFrame, prediction_name: str, attendance_name: str) -> Path:
        prediction_stem = self._safe_stem(prediction_name, "prediction")
        attendance_stem = self._safe_stem(attendance_name, "attendance")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = settings.outputs_dir / f"comparison-{prediction_stem}-vs-{attendance_stem}-{timestamp}.csv"
        comparison_df.to_csv(output_path, index=False)
        return output_path

    def _write_clean_output_csv(self, comparison_df: pd.DataFrame, prediction_name: str, attendance_name: str) -> Path:
        prediction_stem = self._safe_stem(prediction_name, "prediction")
        attendance_stem = self._safe_stem(attendance_name, "attendance")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = settings.outputs_dir / f"comparison-{prediction_stem}-vs-{attendance_stem}-clean-{timestamp}.csv"
        self._clean_comparison_export_df(comparison_df).to_csv(output_path, index=False)
        return output_path

    def _clean_comparison_export_df(self, comparison_df: pd.DataFrame) -> pd.DataFrame:
        column_map = {
            "Employee ID": "employee_id",
            "Date": "date",
            "Attendance In": "attendance_punch_in_time",
            "Attendance Out": "attendance_punch_out_time",
            "Transaction First In": "transaction_first_in_time",
            "Transaction Last Out": "transaction_last_out_time",
            "Attendance Shift": "attendance_truth_shift_family",
            "Predicted Final Shift": "predicted_final_shift",
            "Comparison Result": "comparison_result",
            "Mismatch Reason": "mismatch_reason",
        }
        export_df = pd.DataFrame(index=comparison_df.index)
        for clean_header, source_column in column_map.items():
            if source_column in comparison_df:
                export_df[clean_header] = comparison_df[source_column]
            else:
                export_df[clean_header] = ""
        return export_df

    def _build_punch_series(
        self,
        df: pd.DataFrame,
        direct_column: Optional[str],
        source_column: Optional[str],
        side: str,
    ) -> pd.Series:
        if direct_column:
            direct_values = df[direct_column].map(self._format_punch_value)
        else:
            direct_values = pd.Series("", index=df.index)

        if source_column:
            derived_values = df[source_column].map(lambda value: self._derive_punch_from_text(value, side))
            return direct_values.mask(direct_values.eq(""), derived_values)

        return direct_values

    def _derive_punch_from_text(self, value: Any, side: str) -> str:
        text = self._clean_value(value)
        if not text:
            return ""

        parts = [
            part.strip()
            for part in re.split(r"\s*\|\s*|\s*,\s*|\s*;\s*", text)
            if part.strip()
        ]
        if not parts:
            return ""

        selected = parts[0] if side == "first" else parts[-1]
        return self._format_punch_value(selected)

    def _format_punch_value(self, value: Any) -> str:
        text = self._clean_value(value)
        if not text:
            return ""

        if re.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?", text, flags=re.IGNORECASE):
            parsed_time = pd.to_datetime(text, errors="coerce")
            if pd.notna(parsed_time):
                return parsed_time.strftime("%H:%M:%S")
            return text

        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            if parsed.strftime("%Y-%m-%d") == "1900-01-01":
                return parsed.strftime("%H:%M:%S")
            return parsed.strftime("%Y-%m-%d %H:%M:%S")

        return text

    def _dedupe_keyed_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        rows = rows[rows["employee_id"].ne("") & rows["date"].ne("")].copy()
        return rows.drop_duplicates(subset=["employee_id", "date"], keep="first").reset_index(drop=True)

    def _require_column(self, df: pd.DataFrame, aliases: tuple[str, ...], label: str) -> str:
        column = self._find_column(df.columns, aliases)
        if column is None:
            raise AppError(f"Missing required {label} column.")
        return column

    def _find_column(self, columns: Union[list[str], pd.Index], aliases: tuple[str, ...]) -> Optional[str]:
        normalized_columns: dict[str, str] = {}
        for column in columns:
            normalized_column = self._normalize_column_name(column)
            if normalized_column and normalized_column not in normalized_columns:
                normalized_columns[normalized_column] = str(column)

        for alias in aliases:
            column = normalized_columns.get(self._normalize_column_name(alias))
            if column is not None:
                return column
        return None

    def _normalize_date_series(self, series: pd.Series) -> pd.Series:
        try:
            dates = pd.to_datetime(series, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            return series.map(self._normalize_date_value)

        normalized = dates.dt.strftime("%Y-%m-%d")
        missing = normalized.isna()
        if missing.any():
            normalized.loc[missing] = series.loc[missing].map(self._normalize_date_value)
        return normalized.fillna("")

    def _clean_series(self, series: pd.Series) -> pd.Series:
        cleaned = series.where(series.notna(), "").astype(str).str.strip()
        return cleaned.mask(cleaned.str.upper().isin(NULL_STRINGS), "")

    def _normalize_date_value(self, value: Any) -> str:
        if value is None or pd.isna(value):
            return ""

        if isinstance(value, datetime):
            return value.strftime("%Y-%m-%d")

        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            return parsed.strftime("%Y-%m-%d")

        text_value = str(value).strip()
        if not text_value or text_value.upper() in NULL_STRINGS:
            return ""

        if len(text_value) >= 10:
            parsed = pd.to_datetime(text_value[:10], errors="coerce")
            if pd.notna(parsed):
                return parsed.strftime("%Y-%m-%d")

        return ""

    def _normalize_employee(self, value: Any) -> str:
        clean_value = self._clean_value(value)
        if clean_value.endswith(".0") and clean_value.replace(".", "", 1).isdigit():
            clean_value = clean_value[:-2]
        return clean_value

    def _normalize_attendance_shift_family(self, value: Any) -> str:
        shift = self._normalize_shift(value)
        return ATTENDANCE_SHIFT_FAMILY_MAP.get(shift, shift)

    def _prediction_shift_bucket_series(self, rows: pd.DataFrame) -> pd.Series:
        label = rows["predicted_final_status_label"].fillna("").astype(str).str.upper()
        shift = rows["predicted_shift_norm"].fillna("").astype(str)

        bucket = pd.Series("ASSIGNED", index=rows.index)
        bucket.loc[label.str.contains("WITHHELD", regex=False) | shift.eq("")] = "WITHHELD"
        bucket.loc[label.str.contains("REVIEW", regex=False)] = "REVIEW"
        bucket.loc[label.str.contains("PREDICTED", regex=False)] = "PREDICTED"
        return bucket

    def _prediction_status_category_series(self, rows: pd.DataFrame) -> pd.Series:
        shift = rows["predicted_shift_norm"].fillna("").astype(str)
        label = rows["predicted_final_status_label"].fillna("").astype(str).str.upper()
        day_status = rows["predicted_final_day_status"].fillna("").astype(str).str.upper()

        status = pd.Series("UNKNOWN", index=rows.index)
        status.loc[shift.eq("ABSENT") | label.eq("ABSENT") | day_status.eq("ABSENT")] = "ABSENT"
        status.loc[label.str.contains("WITHHELD", regex=False) | shift.eq("")] = "WITHHELD"
        status.loc[label.str.contains("REVIEW", regex=False)] = "WORKING"
        status.loc[label.str.contains("PREDICTED", regex=False) | day_status.eq("WORKING")] = "WORKING"
        status.loc[shift.eq("WO") | day_status.eq("WO") | self._contains_status_code_series(label, "WO")] = "WO"
        return status

    def _attendance_status_category_series(self, rows: pd.DataFrame) -> pd.Series:
        shift = rows["attendance_truth_shift_family"].fillna("").astype(str).str.upper()
        status_text = rows["attendance_truth_status"].fillna("").astype(str).str.upper()

        status = pd.Series("UNKNOWN", index=rows.index)
        status.loc[shift.ne("")] = "WORKING"
        status.loc[self._contains_status_code_series(status_text, "PR") | self._contains_status_code_series(status_text, "AE")] = "WORKING"
        status.loc[shift.eq("") | shift.isin(NON_WORKING_SHIFT_CODES)] = "NON_WORKING"
        status.loc[shift.eq("ABSENT")] = "ABSENT"
        status.loc[
            self._contains_status_code_series(status_text, "XL")
            | self._contains_status_code_series(status_text, "LEAVE")
            | self._contains_status_code_series(status_text, "L")
            | shift.eq("LEAVE")
        ] = "LEAVE"
        status.loc[
            self._contains_status_code_series(status_text, "AB")
            | self._contains_status_code_series(status_text, "ABSENT")
        ] = "ABSENT"
        status.loc[shift.eq("WO") | self._contains_status_code_series(status_text, "WO")] = "WO"
        return status

    def _prediction_compare_value_series(self, shift: pd.Series, category: pd.Series) -> pd.Series:
        value = shift.fillna("").astype(str)
        category_value = category.fillna("").astype(str).str.upper()
        value = value.mask(category_value.eq("ABSENT"), "ABSENT")
        value = value.mask(category_value.eq("WO"), "WO")
        value = value.mask(category_value.eq("LEAVE"), "LEAVE")
        return value

    def _truth_compare_value_series(self, shift: pd.Series, category: pd.Series) -> pd.Series:
        value = shift.fillna("").astype(str)
        category_value = category.fillna("").astype(str).str.upper()
        value = value.mask(category_value.eq("ABSENT"), "ABSENT")
        value = value.mask(category_value.eq("WO"), "WO")
        value = value.mask(category_value.eq("LEAVE"), "LEAVE")
        value = value.mask(category_value.eq("NON_WORKING") & value.isin(NON_WORKING_SHIFT_CODES), value)
        return value

    def _normalize_shift(self, value: Any) -> str:
        shift = self._clean_value(value).upper()
        if shift in NULL_STRINGS or shift == "UNKNOWN":
            return ""
        if shift.endswith("W") and shift[:-1] in BASE_SHIFT_CODES:
            return shift[:-1]
        return shift

    def _prediction_shift_bucket(self, row: pd.Series) -> str:
        label = self._clean_value(row.get("predicted_final_status_label", "")).upper()
        shift = self._normalize_shift(row.get("predicted_final_shift", ""))

        if "WITHHELD" in label or not shift:
            return "WITHHELD"
        if "REVIEW" in label:
            return "REVIEW"
        if "PREDICTED" in label:
            return "PREDICTED"
        return "ASSIGNED"

    def _prediction_status_category(self, row: pd.Series) -> str:
        shift = self._normalize_shift(row.get("predicted_final_shift", ""))
        label = self._clean_value(row.get("predicted_final_status_label", "")).upper()
        day_status = self._clean_value(row.get("predicted_final_day_status", "")).upper()

        if shift == "ABSENT" or label == "ABSENT" or day_status == "ABSENT":
            return "ABSENT"
        if shift == "WO" or day_status == "WO" or self._contains_status_code(label, "WO"):
            return "WO"
        if "WITHHELD" in label or not shift:
            return "WITHHELD"
        if "REVIEW" in label:
            return "WORKING"
        if "PREDICTED" in label or day_status == "WORKING":
            return "WORKING"
        return "UNKNOWN"

    def _attendance_status_category(self, row: pd.Series) -> str:
        shift = self._clean_value(row.get("attendance_truth_shift_family", "")).upper()
        status = self._clean_value(row.get("attendance_truth_status", "")).upper()

        if shift == "WO" or self._contains_status_code(status, "WO"):
            return "WO"
        if shift == "ABSENT":
            return "ABSENT"
        if any(token in status.split("/") for token in ("AB", "ABSENT")):
            return "ABSENT"
        if any(token in status.split("/") for token in ("XL", "LEAVE", "L")):
            return "LEAVE"
        if not shift or shift in NON_WORKING_SHIFT_CODES:
            return "NON_WORKING"
        if any(token in status.split("/") for token in ("PR", "AE")):
            return "WORKING"
        if shift:
            return "WORKING"
        return "UNKNOWN"

    def _prediction_compare_value(self, shift: str, category: str) -> str:
        category_value = category.upper()
        if category_value in {"ABSENT", "WO", "LEAVE"}:
            return category_value
        return shift

    def _truth_compare_value(self, shift: str, category: str) -> str:
        category_value = category.upper()
        if category_value in {"ABSENT", "WO", "LEAVE"}:
            return category_value
        if category_value == "NON_WORKING" and shift in NON_WORKING_SHIFT_CODES:
            return shift
        return shift

    def _clean_column_name(self, value: Any) -> str:
        return self._clean_value(value)

    def _normalize_column_name(self, value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", self._clean_value(value).lower())

    def _contains_status_code(self, value: Any, code: str) -> bool:
        tokens = re.split(r"[^A-Z0-9]+", self._clean_value(value).upper())
        return code in tokens

    def _contains_status_code_series(self, series: pd.Series, code: str) -> pd.Series:
        pattern = rf"(?:^|[^A-Z0-9]){re.escape(code)}(?:[^A-Z0-9]|$)"
        return series.fillna("").astype(str).str.upper().str.contains(pattern, regex=True)

    def _clean_value(self, value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        text = str(value).strip()
        if text.upper() in NULL_STRINGS:
            return ""
        return text

    def _percent(self, numerator: int, denominator: int) -> float:
        return round((numerator / denominator * 100.0), 2) if denominator else 0.0

    def _safe_stem(self, name: str, fallback: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", Path(name).stem).strip("-") or fallback

    def _sorted_unique_values(self, df: pd.DataFrame, column: str) -> list[str]:
        if column not in df.columns:
            return []

        values = df[column].fillna("").astype(str).str.strip()
        unique_values = {value for value in values if value}
        return sorted(unique_values)

    def _to_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        return json.loads(df.to_json(orient="records", date_format="iso"))


comparison_service = ComparisonService()
