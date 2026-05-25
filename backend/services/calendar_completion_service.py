from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Optional

import pandas as pd

from backend.config import settings


EMPLOYEE_COLUMN_CANDIDATES = ("EmpCode_norm", "EmpCode", "EmployeeCode", "PersonCode", "employee_id")
DATE_COLUMN_CANDIDATES = ("attendance_day", "AttendanceDate", "AccountingDate", "Date", "date")
BACKEND_ROW_SOURCE = "backend_calendar_completion"
MODEL_ROW_SOURCE = "model_prediction"
PLACEHOLDER_VALUE = "-"
DISPLAY_PLACEHOLDER_VALUE = "—"


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date


class HolidayProvider:
    """Placeholder hook for future holiday sources."""

    def is_holiday(self, target_date: date) -> bool:
        return False


class CalendarCompletionService:
    def __init__(self, holiday_provider: Optional[HolidayProvider] = None) -> None:
        self.holiday_provider = holiday_provider or HolidayProvider()

    def complete(self, transaction_df: pd.DataFrame, prediction_df: pd.DataFrame) -> pd.DataFrame:
        if settings.engine_version == "v3_day_first_reader":
            return self._mark_model_rows(prediction_df)

        date_range = self.detect_date_range(transaction_df)
        if date_range is None or prediction_df.empty:
            return self._mark_model_rows(prediction_df)

        employee_column = self._find_column(prediction_df.columns, EMPLOYEE_COLUMN_CANDIDATES) or "EmpCode_norm"
        date_column = self._find_column(prediction_df.columns, DATE_COLUMN_CANDIDATES) or "attendance_day"
        employees = self._transaction_employees(transaction_df)
        if not employees:
            return self._mark_model_rows(prediction_df)

        calendar_df = self.build_employee_date_calendar(employees, date_range)
        missing_df = self.identify_missing_rows(calendar_df, prediction_df, employee_column, date_column)
        generated_df = self.assign_business_rules(missing_df, prediction_df.columns, employee_column, date_column)

        return self.merge_predictions_and_generated(prediction_df, generated_df, employee_column, date_column)

    def detect_date_range(self, transaction_df: pd.DataFrame) -> Optional[DateRange]:
        if "TransactionDateTime" not in transaction_df:
            return None

        transaction_dates = pd.to_datetime(transaction_df["TransactionDateTime"], errors="coerce").dropna()
        if transaction_dates.empty:
            return None

        normalized_dates = transaction_dates.dt.normalize()
        return DateRange(
            start=normalized_dates.min().date(),
            end=normalized_dates.max().date(),
        )

    def build_employee_date_calendar(self, employees: Iterable[str], date_range: DateRange) -> pd.DataFrame:
        dates = pd.date_range(date_range.start, date_range.end, freq="D")
        calendar_index = pd.MultiIndex.from_product(
            [list(employees), dates],
            names=["employee_key", "calendar_date"],
        )
        return calendar_index.to_frame(index=False)

    def identify_missing_rows(
        self,
        calendar_df: pd.DataFrame,
        prediction_df: pd.DataFrame,
        employee_column: str,
        date_column: str,
    ) -> pd.DataFrame:
        existing_keys = self._prediction_keys(prediction_df, employee_column, date_column)
        merged = calendar_df.merge(existing_keys, on=["employee_key", "calendar_date"], how="left", indicator=True)
        return merged[merged["_merge"].eq("left_only")][["employee_key", "calendar_date"]].reset_index(drop=True)

    def assign_business_rules(
        self,
        missing_df: pd.DataFrame,
        prediction_columns: Iterable[str],
        employee_column: str,
        date_column: str,
    ) -> pd.DataFrame:
        columns = self._final_columns(prediction_columns)
        rows: list[dict[str, Any]] = []

        for row in missing_df.itertuples(index=False):
            target_date = pd.Timestamp(row.calendar_date).date()
            completed_row = {column: pd.NA for column in columns}
            completed_row[employee_column] = row.employee_key
            completed_row[date_column] = target_date.isoformat()
            completed_row["weekday_num"] = target_date.weekday()
            completed_row["is_sunday"] = int(target_date.weekday() == 6)
            completed_row["row_source"] = BACKEND_ROW_SOURCE

            self._populate_optional_placeholders(completed_row)
            self._apply_missing_date_rule(completed_row, target_date)
            rows.append(completed_row)

        return pd.DataFrame(rows, columns=columns)

    def merge_predictions_and_generated(
        self,
        prediction_df: pd.DataFrame,
        generated_df: pd.DataFrame,
        employee_column: str,
        date_column: str,
    ) -> pd.DataFrame:
        prediction_rows = self._mark_model_rows(prediction_df)
        final_columns = self._final_columns(prediction_rows.columns)

        prediction_rows = prediction_rows.reindex(columns=final_columns)
        generated_rows = generated_df.reindex(columns=final_columns)
        merged = pd.concat([prediction_rows, generated_rows], ignore_index=True)
        return self._sort_output_rows(merged, employee_column, date_column)

    def _apply_missing_date_rule(self, completed_row: dict[str, Any], target_date: date) -> None:
        completed_row["final_day_status"] = "ABSENT"
        completed_row["final_shift"] = "ABSENT"
        completed_row["shift_status"] = "rule_generated"
        completed_row["final_status_label"] = "ABSENT"
        completed_row["final_message"] = "No transaction found; marked as absent"

    def _populate_optional_placeholders(self, row: dict[str, Any]) -> None:
        for column in (
            "best_shift_candidate",
            "pair_confidence_bucket",
            "prod_model_pred_shift",
            "working_shift_hint",
        ):
            if column in row:
                row[column] = PLACEHOLDER_VALUE

        for column in (
            "window_punch_count",
            "pair_punch_count",
            "pair_start_min",
            "pair_end_min",
            "pair_span_min",
            "pair_next_day_flag",
        ):
            if column in row:
                row[column] = 0

        for column in ("raw_transactions", "pair_preview", "candidate_punch_slice"):
            if column in row:
                row[column] = DISPLAY_PLACEHOLDER_VALUE

        if "valid_pair_found" in row:
            row["valid_pair_found"] = False
        if "prod_model_confidence" in row:
            row["prod_model_confidence"] = pd.NA

    def _transaction_employees(self, transaction_df: pd.DataFrame) -> list[str]:
        if "EmpCode" not in transaction_df:
            return []

        employee_series = transaction_df["EmpCode"].map(self._normalize_employee)
        return sorted(employee for employee in employee_series.dropna().unique().tolist() if employee)

    def _prediction_keys(self, prediction_df: pd.DataFrame, employee_column: str, date_column: str) -> pd.DataFrame:
        if employee_column not in prediction_df or date_column not in prediction_df:
            return pd.DataFrame(columns=["employee_key", "calendar_date"])

        keys = pd.DataFrame(
            {
                "employee_key": prediction_df[employee_column].map(self._normalize_employee),
                "calendar_date": pd.to_datetime(prediction_df[date_column], errors="coerce").dt.normalize(),
            }
        )
        keys = keys[keys["employee_key"].ne("") & keys["calendar_date"].notna()]
        return keys.drop_duplicates()

    def _mark_model_rows(self, prediction_df: pd.DataFrame) -> pd.DataFrame:
        df = prediction_df.copy()
        if "row_source" not in df.columns:
            df["row_source"] = MODEL_ROW_SOURCE
        else:
            df["row_source"] = df["row_source"].fillna(MODEL_ROW_SOURCE).replace("", MODEL_ROW_SOURCE)
        return df

    def _sort_output_rows(self, df: pd.DataFrame, employee_column: str, date_column: str) -> pd.DataFrame:
        if employee_column not in df or date_column not in df:
            return df.reset_index(drop=True)

        sorted_df = df.copy()
        sorted_df["_calendar_sort_date"] = pd.to_datetime(sorted_df[date_column], errors="coerce")
        sorted_df["_calendar_sort_employee"] = sorted_df[employee_column].map(self._normalize_employee)
        sorted_df = sorted_df.sort_values(
            ["_calendar_sort_employee", "_calendar_sort_date"],
            kind="stable",
            na_position="last",
        )
        return sorted_df.drop(columns=["_calendar_sort_employee", "_calendar_sort_date"]).reset_index(drop=True)

    def _final_columns(self, prediction_columns: Iterable[str]) -> list[str]:
        columns = list(prediction_columns)
        if "row_source" not in columns:
            columns.append("row_source")
        if "candidate_punch_slice" not in columns:
            columns.append("candidate_punch_slice")
        return columns

    def _find_column(self, columns: Iterable[str], candidates: tuple[str, ...]) -> Optional[str]:
        column_set = set(columns)
        for candidate in candidates:
            if candidate in column_set:
                return candidate
        return None

    def _normalize_employee(self, value: Any) -> str:
        if pd.isna(value):
            return ""

        text_value = str(value).strip()
        if text_value.endswith(".0"):
            return text_value[:-2]
        return text_value


calendar_completion_service = CalendarCompletionService()
