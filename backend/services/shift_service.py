import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from backend.config import settings
from backend.services.file_service import AppError
from shift_engine_package_v3_day_first_reader.inference import CONFIG as ENGINE_CONFIG
from shift_engine_package_v3_day_first_reader.inference import LABEL_CLASSES, run_inference_debug


REQUIRED_COLUMNS = ("EmpCode", "TransactionDateTime", "ReaderNumber")
EMPLOYEE_ID_ALIASES = (
    "EmpCode",
    "EmployeeCode",
    "PersonCode",
    "PersonCode_norm",
    "EmpCode_norm",
)
TRANSACTION_DATETIME_ALIASES = (
    "TransactionDateTime",
    "PunchDateTime",
    "DateTime",
    "TxnDateTime",
    "Transaction_Time",
    "PunchTime",
)
READER_NUMBER_ALIASES = (
    "ReaderNumber",
    "Reader_Number",
    "reader_number",
)
SUPPORTED_ALIASES = {
    "EmpCode": list(EMPLOYEE_ID_ALIASES),
    "TransactionDateTime": list(TRANSACTION_DATETIME_ALIASES),
    "ReaderNumber": list(READER_NUMBER_ALIASES),
}

logger = logging.getLogger("shift_prediction")


@dataclass
class PredictionResult:
    file_name: str
    output_file_name: str
    output_path: Path
    clean_output_file_name: str
    clean_output_path: Path
    generated_at: datetime
    summary: dict[str, int]
    columns: list[str]
    records: list[dict[str, Any]]


@dataclass
class PredictionCacheEntry:
    file_hash: str
    file_name: str
    output_file_name: str
    output_path: Path
    clean_output_file_name: str
    clean_output_path: Path
    generated_at: datetime


class ShiftPredictionService:
    def __init__(self) -> None:
        self.engine_config = ENGINE_CONFIG
        self.model_classes = list(LABEL_CLASSES)
        self._prediction_cache: dict[str, PredictionCacheEntry] = {}
        self._latest_employee_history: dict[str, dict[str, Any]] = {}

    def predict(
        self,
        file_path: Path,
        original_name: str,
        *,
        include_records: bool = True,
        use_cache: bool = False,
    ) -> PredictionResult:
        total_started = perf_counter()
        timings: dict[str, float] = {}
        logger.info("Prediction upload received: %s", original_name)

        started = perf_counter()
        file_hash = self._hash_file(file_path)
        timings["file_hash"] = perf_counter() - started

        if use_cache:
            cached_result = self._cached_prediction_result(file_hash, include_records=include_records)
            if cached_result is not None:
                timings["total"] = perf_counter() - total_started
                logger.info(
                    "Prediction cache hit for %s: file_hash=%.3fs total=%.3fs output=%s",
                    original_name,
                    timings["file_hash"],
                    timings["total"],
                    cached_result.output_file_name,
                )
                return cached_result

        try:
            started = perf_counter()
            transaction_df = self._read_transaction_file(file_path)
            timings["file_load"] = perf_counter() - started
        except AppError:
            raise
        except ValueError as exc:
            raise AppError(str(exc)) from exc
        except Exception as exc:
            raise AppError(
                "The uploaded file could not be read. Check that the file is a valid Excel or CSV document."
            ) from exc

        started = perf_counter()
        transaction_df, detected_columns, mapped_columns = self._normalize_transaction_df(transaction_df)
        timings["transaction_normalization"] = perf_counter() - started
        logger.info("Detected columns for %s: %s", original_name, detected_columns)
        logger.info("Mapped columns for %s: %s", original_name, mapped_columns)

        started = perf_counter()
        self._validate_transaction_df(transaction_df, detected_columns)
        timings["validation"] = perf_counter() - started

        try:
            started = perf_counter()
            normalized_input_path = self._write_normalized_engine_input(transaction_df, original_name)
            try:
                result_df = run_inference_debug(str(normalized_input_path))
            finally:
                normalized_input_path.unlink(missing_ok=True)
            result_df = self._normalize_engine_result_df(result_df)
            timings["engine_prediction"] = perf_counter() - started
        except Exception as exc:
            raise AppError("Prediction failed while processing the transaction file.", status_code=500) from exc

        started = perf_counter()
        result_df = self._normalize_pair_validity(result_df)
        timings["pair_validity_normalization"] = perf_counter() - started

        started = perf_counter()
        result_df = self._add_prediction_mismatch_fields(result_df)
        timings["prediction_mismatch"] = perf_counter() - started

        if result_df.empty:
            raise AppError("No prediction rows were generated from the uploaded file.")

        started = perf_counter()
        result_df = self._reorder_output_columns(result_df)
        timings["calendar_completion"] = perf_counter() - started

        started = perf_counter()
        output_path = self._write_output_csv(result_df, original_name)
        clean_output_path = self._write_clean_output_csv(result_df, original_name)
        timings["csv_generation"] = perf_counter() - started

        started = perf_counter()
        summary = self._build_summary(result_df)
        records = self._to_records(result_df) if include_records else []
        timings["response_serialization"] = perf_counter() - started
        timings["total"] = perf_counter() - total_started
        self._latest_employee_history = self._build_employee_history(result_df)

        logger.info(
            "Prediction timing summary for %s: file_load=%.3fs normalize=%.3fs validation=%.3fs engine=%.3fs pair_validity=%.3fs calendar_completion=%.3fs csv=%.3fs response=%.3fs total=%.3fs rows=%s",
            original_name,
            timings.get("file_load", 0.0),
            timings.get("transaction_normalization", 0.0),
            timings.get("validation", 0.0),
            timings.get("engine_prediction", 0.0),
            timings.get("pair_validity_normalization", 0.0),
            timings.get("calendar_completion", 0.0),
            timings.get("csv_generation", 0.0),
            timings.get("response_serialization", 0.0),
            timings.get("total", 0.0),
            len(result_df),
        )

        if file_hash:
            self._prediction_cache[file_hash] = PredictionCacheEntry(
                file_hash=file_hash,
                file_name=original_name,
                output_file_name=output_path.name,
                output_path=output_path,
                clean_output_file_name=clean_output_path.name,
                clean_output_path=clean_output_path,
                generated_at=datetime.now(),
            )

        return PredictionResult(
            file_name=original_name,
            output_file_name=output_path.name,
            output_path=output_path,
            clean_output_file_name=clean_output_path.name,
            clean_output_path=clean_output_path,
            generated_at=datetime.now(),
            summary=summary,
            columns=result_df.columns.tolist(),
            records=records,
        )

    def predict_for_comparison(self, file_path: Path, original_name: str) -> PredictionResult:
        return self.predict(
            file_path=file_path,
            original_name=original_name,
            include_records=False,
            use_cache=True,
        )

    def manual_predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        employee_id = str(payload.get("employee_id") or "").strip()
        if not employee_id:
            raise AppError("Employee ID is required.")

        attendance_day = self._parse_manual_date(payload.get("date"))
        transaction_df, raw_transactions = self._build_manual_transaction_df(
            employee_id=employee_id,
            attendance_day=attendance_day,
            punch_in=payload.get("punch_in"),
            punch_out=payload.get("punch_out"),
            extra_punches=payload.get("extra_punches"),
        )

        try:
            normalized_input_path = self._write_normalized_engine_input(transaction_df, f"manual-{employee_id}.csv")
            try:
                result_df = run_inference_debug(str(normalized_input_path))
            finally:
                normalized_input_path.unlink(missing_ok=True)
            result_df = self._normalize_engine_result_df(result_df)
            result_df = self._normalize_pair_validity(result_df)
        except Exception as exc:
            raise AppError("Manual prediction failed while processing the supplied punches.", status_code=500) from exc

        if result_df.empty:
            raise AppError("No prediction row was generated from the supplied manual punches.")

        manual_row = self._select_manual_attendance_row(result_df, attendance_day)
        row_dict = self._json_safe_dict(manual_row.to_dict())
        history = self._lookup_employee_history(employee_id)

        return self._build_manual_prediction_response(
            employee_id=employee_id,
            attendance_day=attendance_day,
            row=row_dict,
            history=history,
            raw_transactions=raw_transactions,
            note=payload.get("note"),
        )

    def _cached_prediction_result(self, file_hash: str, *, include_records: bool) -> PredictionResult | None:
        entry = self._prediction_cache.get(file_hash)
        if entry is None or not entry.output_path.exists() or not entry.clean_output_path.exists():
            return None

        try:
            result_df = pd.read_csv(entry.output_path, low_memory=False)
        except Exception:
            logger.exception("Failed to read cached prediction output: %s", entry.output_path)
            return None

        result_df = self._normalize_engine_result_df(result_df)
        result_df = self._normalize_pair_validity(result_df)
        if "prediction_mismatch_status" not in result_df or "prediction_mismatch_message" not in result_df:
            result_df = self._add_prediction_mismatch_fields(result_df)

        return PredictionResult(
            file_name=entry.file_name,
            output_file_name=entry.output_file_name,
            output_path=entry.output_path,
            clean_output_file_name=entry.clean_output_file_name,
            clean_output_path=entry.clean_output_path,
            generated_at=entry.generated_at,
            summary=self._build_summary(result_df),
            columns=result_df.columns.tolist(),
            records=self._to_records(result_df) if include_records else [],
        )

    def _hash_file(self, file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _read_transaction_file(self, file_path: Path) -> pd.DataFrame:
        suffix = file_path.suffix.lower()
        if suffix in {".csv", ".txt", ".tsv"}:
            return pd.read_csv(file_path, sep=None, engine="python")
        if suffix in {".xls", ".xlsx"}:
            try:
                df = pd.read_csv(file_path, sep="\t", low_memory=False)
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass
            return pd.read_excel(file_path)
        raise AppError(f"Unsupported file type: {suffix}")

    def _write_normalized_engine_input(self, transaction_df: pd.DataFrame, original_name: str) -> Path:
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(original_name).stem).strip("-") or "engine-input"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        target_path = settings.uploads_dir / f"{safe_stem}-v3-day-first-normalized-{timestamp}.csv"
        engine_input = transaction_df.copy()
        engine_input.to_csv(target_path, index=False)
        return target_path

    def _normalize_engine_result_df(self, result_df: pd.DataFrame) -> pd.DataFrame:
        df = result_df.copy()
        rename_map = {
            "PersonCode": "EmpCode",
            "PersonCode_norm": "EmpCode_norm",
            "Employee ID": "EmpCode_norm",
            "Date": "attendance_day",
            "Final Shift": "final_shift",
            "Final Status Label": "final_status_label",
            "Final Day Status": "final_day_status",
            "Shift Status": "shift_status",
            "Final Message": "final_message",
            "Mismatch Status": "prediction_mismatch_status",
            "Mismatch Message": "prediction_mismatch_message",
            "Same-Day Punch Count": "txn_punch_count",
            "Punch In": "punch_in_time",
            "Punch Out": "punch_out_time",
            "Valid Reader Pair Found": "valid_reader_pair_found",
            "Raw Transactions": "raw_transactions",
            "Candidate Punch Slice": "pair_preview",
            "Model Predicted Shift": "prod_model_pred_shift",
            "Model Confidence": "prod_model_confidence",
            "Decision Reason": "decision_reason",
        }
        df = df.rename(columns={source: target for source, target in rename_map.items() if source in df.columns})

        if "EmpCode_norm" not in df.columns and "EmpCode" in df.columns:
            df["EmpCode_norm"] = df["EmpCode"].astype(str).str.strip()
        if "EmpCode" not in df.columns and "EmpCode_norm" in df.columns:
            df["EmpCode"] = df["EmpCode_norm"]
        if "punch_in_time" not in df.columns and "pair_first_punch" in df.columns:
            df["punch_in_time"] = df["pair_first_punch"]
        if "punch_out_time" not in df.columns and "pair_last_punch" in df.columns:
            df["punch_out_time"] = df["pair_last_punch"]
        if "valid_reader_pair_found" not in df.columns and "reader_pair_found_flag" in df.columns:
            df["valid_reader_pair_found"] = pd.to_numeric(df["reader_pair_found_flag"], errors="coerce").fillna(0).astype(int).eq(1)
        if "valid_pair_found" not in df.columns and "valid_reader_pair_found" in df.columns:
            df["valid_pair_found"] = df["valid_reader_pair_found"]
        if "txn_punch_count" not in df.columns:
            df["txn_punch_count"] = 0

        for column in ("attendance_day", "punch_in_time", "punch_out_time", "pair_first_punch", "pair_last_punch"):
            if column in df.columns:
                parsed = pd.to_datetime(df[column], errors="coerce")
                if column == "attendance_day":
                    df[column] = parsed.dt.strftime("%Y-%m-%d").fillna("")
                else:
                    df[column] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")

        if "prod_model_confidence" in df.columns:
            confidence = pd.to_numeric(df["prod_model_confidence"], errors="coerce")
            if confidence.dropna().gt(1).any():
                confidence = confidence / 100.0
            df["prod_model_confidence"] = confidence

        return df

    def _normalize_transaction_df(
        self,
        transaction_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str], dict[str, str | None]]:
        detected_columns = transaction_df.columns.tolist()
        employee_column = self._find_first_matching_column(detected_columns, EMPLOYEE_ID_ALIASES)
        datetime_column = self._find_first_matching_column(detected_columns, TRANSACTION_DATETIME_ALIASES)
        reader_number_column = self._find_first_matching_column(detected_columns, READER_NUMBER_ALIASES)

        normalized_df = transaction_df.copy()
        rename_map: dict[str, str] = {}

        if employee_column and employee_column != "EmpCode":
            rename_map[employee_column] = "EmpCode"

        if datetime_column and datetime_column != "TransactionDateTime":
            rename_map[datetime_column] = "TransactionDateTime"

        if reader_number_column and reader_number_column != "ReaderNumber":
            rename_map[reader_number_column] = "ReaderNumber"

        if rename_map:
            normalized_df = normalized_df.rename(columns=rename_map)

        mapped_columns = {
            "EmpCode": employee_column,
            "TransactionDateTime": datetime_column,
            "ReaderNumber": reader_number_column,
        }
        return normalized_df, detected_columns, mapped_columns

    def _find_first_matching_column(self, detected_columns: list[str], aliases: tuple[str, ...]) -> str | None:
        normalized_columns = {
            re.sub(r"[^a-z0-9]+", "", str(column).lower()): column
            for column in detected_columns
        }
        for alias in aliases:
            normalized_alias = re.sub(r"[^a-z0-9]+", "", alias.lower())
            if normalized_alias in normalized_columns:
                return str(normalized_columns[normalized_alias])
        return None

    def _validate_transaction_df(self, transaction_df: pd.DataFrame, detected_columns: list[str]) -> None:
        if transaction_df.empty:
            raise AppError("The uploaded file is empty.")

        missing_columns = [column for column in REQUIRED_COLUMNS if column not in transaction_df.columns]
        if missing_columns:
            if missing_columns == ["ReaderNumber"]:
                raise AppError(
                    "Missing required column: ReaderNumber. V3 day-first reader engine requires ReaderNumber where 1 = IN and 2 = OUT.",
                    extra={
                        "detected_columns": detected_columns,
                        "supported_aliases": SUPPORTED_ALIASES,
                    },
                )
            missing = ", ".join(missing_columns)
            raise AppError(
                f"Missing required column(s): {missing}.",
                extra={
                    "detected_columns": detected_columns,
                    "supported_aliases": SUPPORTED_ALIASES,
                },
            )

        valid_datetimes = pd.to_datetime(transaction_df["TransactionDateTime"], errors="coerce").notna()
        if not valid_datetimes.any():
            raise AppError("No valid `TransactionDateTime` values were found in the uploaded file.")

        valid_employees = transaction_df["EmpCode"].astype(str).str.strip().ne("")
        if not valid_employees.any():
            raise AppError("No valid `EmpCode` values were found in the uploaded file.")

        valid_reader_numbers = pd.to_numeric(transaction_df["ReaderNumber"], errors="coerce").isin([1, 2])
        if not valid_reader_numbers.any():
            raise AppError("No valid `ReaderNumber` values were found. V3 day-first reader engine requires ReaderNumber where 1 = IN and 2 = OUT.")

    def _write_output_csv(self, result_df: pd.DataFrame, original_name: str) -> Path:
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(original_name).stem).strip("-") or "prediction"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = settings.outputs_dir / f"{safe_stem}-predictions-{timestamp}.csv"
        result_df.to_csv(output_path, index=False)
        return output_path

    def _write_clean_output_csv(self, result_df: pd.DataFrame, original_name: str) -> Path:
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(original_name).stem).strip("-") or "prediction"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = settings.outputs_dir / f"{safe_stem}-predictions-clean-{timestamp}.csv"
        self._clean_prediction_export_df(result_df).to_csv(output_path, index=False)
        return output_path

    def _clean_prediction_export_df(self, result_df: pd.DataFrame) -> pd.DataFrame:
        column_map = {
            "Employee ID": "EmpCode_norm",
            "Date": "attendance_day",
            "Final Shift": "final_shift",
            "Final Status Label": "final_status_label",
            "Final Day Status": "final_day_status",
            "Shift Status": "shift_status",
            "Final Message": "final_message",
            "Same-Day Punch Count": "txn_punch_count",
            "Punch In": "punch_in_time",
            "Punch Out": "punch_out_time",
            "Valid Reader Pair Found": "valid_reader_pair_found",
            "Raw Transactions": "raw_transactions",
            "Candidate Punch Slice": "pair_preview",
            "Decision Reason": "decision_reason",
        }
        export_df = self._export_with_business_headers(result_df, column_map)
        absent_mask = export_df["Final Status Label"].astype(str).str.upper().eq("ABSENT")
        export_df.loc[absent_mask, "Same-Day Punch Count"] = 0
        for column in ("Punch In", "Punch Out", "Raw Transactions"):
            export_df.loc[absent_mask, column] = ""
        if "Punch In" in export_df:
            export_df["Punch In"] = export_df["Punch In"].map(self._clean_export_text)
        if "Punch Out" in export_df:
            export_df["Punch Out"] = export_df["Punch Out"].map(self._clean_export_text)
        return export_df

    def _export_with_business_headers(self, source_df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
        export_df = pd.DataFrame(index=source_df.index)
        for clean_header, source_column in column_map.items():
            if source_column in source_df:
                export_df[clean_header] = source_df[source_column]
            else:
                export_df[clean_header] = ""
        return export_df

    def _clean_punch_export_series(self, result_df: pd.DataFrame, punch_side: str) -> pd.Series:
        source_column = "punch_in_time" if punch_side == "in" else "punch_out_time"
        if source_column in result_df:
            direct_values = result_df[source_column].map(self._clean_export_text)
        else:
            direct_values = pd.Series("", index=result_df.index)

        derived_values = result_df.apply(
            lambda row: self._derive_punch_from_candidate_slice(row, punch_side),
            axis=1,
        )
        values = direct_values.mask(direct_values.eq(""), derived_values)
        return values.mask(values.eq(""), "—")

    def _derive_punch_from_candidate_slice(self, row: pd.Series, punch_side: str) -> str:
        valid_pair_found = self._coerce_bool(row.get("valid_pair_found"))
        pair_preview = self._clean_export_text(row.get("pair_preview"))
        if not valid_pair_found or pair_preview in {"", "-", "—"}:
            return "—"

        punches = [part.strip() for part in pair_preview.split("|") if part.strip()]
        if not punches:
            return "—"

        return punches[0] if punch_side == "in" else punches[-1]

    def _clean_export_text(self, value: Any) -> str:
        if value is None or pd.isna(value):
            return ""
        text = str(value).strip()
        if text.lower() in {"", "nan", "none", "null", "nat"}:
            return ""
        return text

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None or pd.isna(value):
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"true", "1", "yes", "y"}

    def _build_summary(self, result_df: pd.DataFrame) -> dict[str, int]:
        labels = result_df["final_status_label"].astype(str).str.upper() if "final_status_label" in result_df else pd.Series([], dtype=str)
        status_counts = labels.value_counts()

        return {
            "total_rows": int(len(result_df)),
            "SHIFT_PREDICTED": int(status_counts.get("SHIFT_PREDICTED", 0)),
            "SHIFT_REVIEW": int(status_counts.get("SHIFT_REVIEW", 0)),
            "WITHHELD": int(status_counts.get("WITHHELD", 0)),
            "ABSENT": int(status_counts.get("ABSENT", 0)),
            "WO": int((result_df["final_shift"].astype(str).str.upper().eq("WO")).sum()) if "final_shift" in result_df else 0,
            "WO_REVIEW": int(status_counts.get("WO_REVIEW", 0)),
        }

    def _normalize_pair_validity(self, result_df: pd.DataFrame) -> pd.DataFrame:
        df = result_df.copy()
        if "reader_pair_found_flag" in df:
            valid_pair_found = pd.to_numeric(df["reader_pair_found_flag"], errors="coerce").fillna(0).astype(int).eq(1)
        elif "valid_reader_pair_found" in df:
            valid_pair_found = df["valid_reader_pair_found"].map(self._coerce_bool)
        else:
            valid_pair_found = pd.Series(False, index=df.index)

        df["valid_reader_pair_found"] = valid_pair_found.astype(bool)
        df["valid_pair_found"] = valid_pair_found.astype(bool)

        return df

    def _add_prediction_mismatch_fields(self, result_df: pd.DataFrame) -> pd.DataFrame:
        df = result_df.copy()
        statuses: list[str] = []
        messages: list[str] = []

        for _, row in df.iterrows():
            status, message = self._prediction_mismatch_for_row(row)
            statuses.append(status)
            messages.append(message)

        df["prediction_mismatch_status"] = statuses
        df["prediction_mismatch_message"] = messages
        return df

    def _prediction_mismatch_for_row(self, row: pd.Series) -> tuple[str, str]:
        final_status_label = str(row.get("final_status_label") or "").strip().upper()
        final_message = self._clean_export_text(row.get("final_message"))
        model_shift = self._clean_shift_value(row.get("prod_model_pred_shift"))
        final_shift = self._clean_shift_value(row.get("final_shift"))

        if final_status_label in {"ABSENT", "WO_REVIEW", "WITHHELD"}:
            return (
                "NOT_APPLICABLE",
                final_message or f"{final_status_label} row; model-vs-final shift mismatch is not applicable.",
            )

        if not model_shift or not final_shift:
            return (
                "NOT_APPLICABLE",
                final_message or "Model or final shift is unavailable; mismatch is not applicable.",
            )

        if model_shift != final_shift:
            return (
                "MISMATCH",
                f"Model predicted {model_shift}, but final shift is {final_shift}.",
            )

        return (
            "MATCH",
            f"Model predicted shift matches final shift: {final_shift}.",
        )

    def _reorder_output_columns(self, result_df: pd.DataFrame) -> pd.DataFrame:
        move_after_column = "final_status_label"
        moved_columns = ["valid_reader_pair_found", "txn_punch_count"]

        if move_after_column not in result_df.columns:
            return result_df

        current_columns = [
            column
            for column in result_df.columns
            if column not in moved_columns
        ]
        existing_moved_columns = [
            column
            for column in moved_columns
            if column in result_df.columns
        ]

        insert_at = current_columns.index(move_after_column) + 1
        reordered_columns = (
            current_columns[:insert_at]
            + existing_moved_columns
            + current_columns[insert_at:]
        )
        return result_df.loc[:, reordered_columns]

    def _to_records(self, result_df: pd.DataFrame) -> list[dict[str, Any]]:
        return json.loads(result_df.to_json(orient="records", date_format="iso"))

    def _parse_manual_date(self, value: Any) -> pd.Timestamp:
        if value is None or str(value).strip() == "":
            raise AppError("Date is required.")

        parsed = pd.to_datetime(str(value).strip(), errors="coerce")
        if pd.isna(parsed):
            raise AppError("Date must be a valid date.")

        return pd.Timestamp(parsed).normalize()

    def _build_manual_transaction_df(
        self,
        *,
        employee_id: str,
        attendance_day: pd.Timestamp,
        punch_in: Any,
        punch_out: Any,
        extra_punches: Any,
    ) -> tuple[pd.DataFrame, list[str]]:
        if punch_in is None or str(punch_in).strip() == "":
            raise AppError("Punch In time is required.")
        if punch_out is None or str(punch_out).strip() == "":
            raise AppError("Punch Out time is required.")

        punch_in_ts, _ = self._parse_manual_punch_time(
            punch_in,
            attendance_day,
            field_label="Punch In",
        )
        punch_out_ts, punch_out_time_only = self._parse_manual_punch_time(
            punch_out,
            attendance_day,
            field_label="Punch Out",
        )
        if punch_out_time_only and punch_out_ts <= punch_in_ts:
            punch_out_ts += pd.Timedelta(days=1)

        parsed_punches: list[tuple[pd.Timestamp, int]] = [(punch_in_ts, 1), (punch_out_ts, 2)]
        for extra_punch in self._split_extra_punches(extra_punches):
            extra_ts, extra_time_only = self._parse_manual_punch_time(
                extra_punch,
                attendance_day,
                field_label="Extra punch",
            )
            if extra_time_only and extra_ts < punch_in_ts:
                extra_ts += pd.Timedelta(days=1)
            parsed_punches.append((extra_ts, 2 if extra_ts >= punch_out_ts else 1))

        deduped_punches = sorted(set((pd.Timestamp(punch).to_pydatetime(), reader_number) for punch, reader_number in parsed_punches))
        punch_window_hours = int(self.engine_config.get("window_hours_after", 48))
        window_end = attendance_day + pd.Timedelta(hours=punch_window_hours)
        in_window_punches = [
            (pd.Timestamp(punch), reader_number)
            for punch, reader_number in deduped_punches
            if attendance_day <= pd.Timestamp(punch) < window_end
        ]

        if len(in_window_punches) < 2:
            raise AppError(
                "No valid in-out pair found in the manual prediction window. "
                "Use punches on the selected date or within the engine window after that date."
            )

        transaction_df = pd.DataFrame(
            {
                "EmpCode": [employee_id] * len(in_window_punches),
                "TransactionDateTime": [punch for punch, _ in in_window_punches],
                "ReaderNumber": [reader_number for _, reader_number in in_window_punches],
                "ReaderId": [None] * len(in_window_punches),
                "ReasonCode": [None] * len(in_window_punches),
                "TransactionCode": [None] * len(in_window_punches),
            }
        )
        raw_transactions = [
            f"{pd.Timestamp(punch).strftime('%Y-%m-%d %H:%M:%S')} (ReaderNumber={reader_number})"
            for punch, reader_number in in_window_punches
        ]
        return transaction_df, raw_transactions

    def _parse_manual_punch_time(
        self,
        value: Any,
        attendance_day: pd.Timestamp,
        *,
        field_label: str,
    ) -> tuple[pd.Timestamp, bool]:
        text = str(value or "").strip()
        if not text:
            raise AppError(f"{field_label} time is required.")

        time_only = self._is_time_only_value(text)
        parse_text = f"{attendance_day.date()} {text}" if time_only else text
        parsed = pd.to_datetime(parse_text, errors="coerce")
        if pd.isna(parsed):
            raise AppError(f"{field_label} must be a valid time or datetime.")

        return pd.Timestamp(parsed).replace(tzinfo=None), time_only

    def _is_time_only_value(self, value: str) -> bool:
        return bool(
            re.fullmatch(
                r"\s*\d{1,2}(:\d{2}){0,2}(\s*[AP]M)?\s*",
                value,
                flags=re.IGNORECASE,
            )
        )

    def _split_extra_punches(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            raw_values = value
        else:
            raw_values = re.split(r"[\n,]+", str(value))

        return [str(item).strip() for item in raw_values if str(item).strip()]

    def _select_manual_attendance_row(self, result_df: pd.DataFrame, attendance_day: pd.Timestamp) -> pd.Series:
        if "attendance_day" not in result_df.columns:
            return result_df.iloc[0]

        days = pd.to_datetime(result_df["attendance_day"], errors="coerce").dt.normalize()
        matching_rows = result_df[days.eq(attendance_day)]
        if matching_rows.empty:
            return result_df.iloc[0]

        return matching_rows.iloc[0]

    def _build_employee_history(self, result_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        if result_df.empty or "EmpCode_norm" not in result_df.columns or "final_shift" not in result_df.columns:
            return {}

        history: dict[str, dict[str, Any]] = {}
        df = result_df.copy()
        df["EmpCode_norm"] = df["EmpCode_norm"].astype(str).str.strip()
        df["final_shift"] = df["final_shift"].astype(str).str.strip()

        usable_mask = df["EmpCode_norm"].ne("") & df["final_shift"].ne("")
        usable_mask &= ~df["final_shift"].str.lower().isin({"unknown", "nan", "none", "null", "wo", "-"})
        if "final_status_label" in df.columns:
            usable_mask &= df["final_status_label"].astype(str).str.upper().eq("SHIFT_PREDICTED")

        usable_df = df[usable_mask]
        for employee_id, group in usable_df.groupby("EmpCode_norm", sort=False):
            shift_counts = group["final_shift"].value_counts()
            if shift_counts.empty:
                continue

            regular_shift = str(shift_counts.index[0])
            support = int(shift_counts.sum())
            top_count = int(shift_counts.iloc[0])
            confidence = float(top_count / support) if support else 0.0
            history[employee_id] = {
                "regular_shift": regular_shift,
                "confidence": confidence,
                "consistency": f"{top_count}/{support}",
                "support": support,
                "source": "latest_prediction_cache",
                "strong": bool(support >= 3 and confidence >= 0.70),
            }

        return history

    def _lookup_employee_history(self, employee_id: str) -> dict[str, Any] | None:
        normalized_employee_id = str(employee_id).strip()
        if normalized_employee_id in self._latest_employee_history:
            return self._latest_employee_history[normalized_employee_id]

        saved_history = self._load_saved_employee_history()
        return saved_history.get(normalized_employee_id)

    def _load_saved_employee_history(self) -> dict[str, dict[str, Any]]:
        candidate_paths = [
            settings.engine_artifacts_dir / "employee_history.json",
            settings.engine_artifacts_dir / "employee_shift_history.json",
            settings.engine_artifacts_dir / "employee_history.csv",
            settings.engine_artifacts_dir / "employee_shift_history.csv",
            settings.outputs_dir / "employee_history.json",
            settings.outputs_dir / "employee_shift_history.json",
            settings.outputs_dir / "employee_history.csv",
            settings.outputs_dir / "employee_shift_history.csv",
        ]

        for path in candidate_paths:
            if not path.exists():
                continue

            try:
                if path.suffix.lower() == ".json":
                    return self._load_employee_history_json(path)
                if path.suffix.lower() == ".csv":
                    return self._load_employee_history_csv(path)
            except Exception:
                logger.exception("Failed to load saved employee history artifact: %s", path)

        return {}

    def _load_employee_history_json(self, path: Path) -> dict[str, dict[str, Any]]:
        with path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)

        if isinstance(payload, dict):
            rows = [
                {"employee_id": employee_id, **record}
                for employee_id, record in payload.items()
                if isinstance(record, dict)
            ]
        elif isinstance(payload, list):
            rows = [record for record in payload if isinstance(record, dict)]
        else:
            return {}

        return self._history_from_rows(pd.DataFrame(rows), source=f"saved_artifact:{path.name}")

    def _load_employee_history_csv(self, path: Path) -> dict[str, dict[str, Any]]:
        return self._history_from_rows(pd.read_csv(path), source=f"saved_artifact:{path.name}")

    def _history_from_rows(self, rows_df: pd.DataFrame, *, source: str) -> dict[str, dict[str, Any]]:
        if rows_df.empty:
            return {}

        columns = rows_df.columns.tolist()
        employee_column = self._find_first_matching_column(
            columns,
            ("EmpCode_norm", "EmpCode", "employee_id", "Employee ID", "EmployeeCode"),
        )
        regular_shift_column = self._find_first_matching_column(
            columns,
            ("regular_shift", "historical_regular_shift", "final_shift", "shift", "WorkingShift"),
        )

        if employee_column is None or regular_shift_column is None:
            return {}

        confidence_column = self._find_first_matching_column(
            columns,
            ("history_confidence", "confidence", "consistency_score"),
        )
        consistency_column = self._find_first_matching_column(
            columns,
            ("history_consistency", "consistency"),
        )
        support_column = self._find_first_matching_column(
            columns,
            ("history_support", "support", "count", "shift_count"),
        )

        normalized = rows_df.copy()
        normalized[employee_column] = normalized[employee_column].astype(str).str.strip()
        normalized[regular_shift_column] = normalized[regular_shift_column].astype(str).str.strip()
        if normalized[employee_column].duplicated().any():
            grouped_history: dict[str, dict[str, Any]] = {}
            for employee_id, group in normalized.groupby(employee_column, sort=False):
                grouped_history.update(self._build_employee_history(group.rename(columns={
                    employee_column: "EmpCode_norm",
                    regular_shift_column: "final_shift",
                })))
                if employee_id in grouped_history:
                    grouped_history[employee_id]["source"] = source
            return grouped_history

        history: dict[str, dict[str, Any]] = {}
        for _, row in normalized.iterrows():
            employee_id = str(row.get(employee_column) or "").strip()
            regular_shift = str(row.get(regular_shift_column) or "").strip()
            if not employee_id or not regular_shift or regular_shift.lower() in {"unknown", "nan", "none", "null", "wo", "-"}:
                continue

            confidence = self._optional_float(row.get(confidence_column)) if confidence_column else None
            support = self._optional_int(row.get(support_column)) if support_column else None
            consistency = str(row.get(consistency_column)).strip() if consistency_column and pd.notna(row.get(consistency_column)) else None
            strong = bool(
                (confidence is not None and confidence >= 0.70)
                and (support is None or support >= 3)
            )

            history[employee_id] = {
                "regular_shift": regular_shift,
                "confidence": confidence,
                "consistency": consistency,
                "support": support,
                "source": source,
                "strong": strong,
            }

        return history

    def _build_manual_prediction_response(
        self,
        *,
        employee_id: str,
        attendance_day: pd.Timestamp,
        row: dict[str, Any],
        history: dict[str, Any] | None,
        raw_transactions: list[str],
        note: Any,
    ) -> dict[str, Any]:
        model_shift = self._clean_shift_value(row.get("prod_model_pred_shift"))
        model_confidence = self._optional_float(row.get("prod_model_confidence"))
        engine_final_shift = self._clean_shift_value(row.get("final_shift")) or "unknown"

        historical_shift = "unknown"
        history_confidence = None
        history_consistency = None
        history_support = None
        history_source = None

        if history:
            historical_shift = self._clean_shift_value(history.get("regular_shift")) or "unknown"
            history_confidence = self._optional_float(history.get("confidence"))
            history_consistency = history.get("consistency")
            history_support = self._optional_int(history.get("support"))
            history_source = history.get("source")

        if historical_shift != "unknown":
            final_shift = historical_shift
            if model_shift and model_shift == historical_shift:
                status = "predicted"
                message = "Manual punch timing matches employee regular shift"
            else:
                status = "review"
                message = "Manual punch timing differs from employee regular shift; regular shift retained for review"
        else:
            final_shift = model_shift or engine_final_shift
            status = "review"
            message = "No employee history available; recommendation is based on manual punch timing only"

        return {
            "employee_id": employee_id,
            "date": attendance_day.date().isoformat(),
            "model_predicted_shift": model_shift,
            "model_confidence": model_confidence,
            "historical_regular_shift": historical_shift,
            "history_confidence": history_confidence,
            "history_consistency": history_consistency,
            "history_support": history_support,
            "history_source": history_source,
            "final_recommended_shift": final_shift or "unknown",
            "status": status,
            "message": message,
            "raw_transactions_used": raw_transactions,
            "raw_transactions_text": " | ".join(raw_transactions),
            "note": str(note).strip() if note is not None and str(note).strip() else None,
            "engine_row": row,
        }

    def _row_has_valid_pair(self, row: dict[str, Any]) -> bool:
        if "valid_reader_pair_found" in row:
            return self._coerce_bool(row.get("valid_reader_pair_found"))
        if "reader_pair_found_flag" in row:
            return self._coerce_bool(row.get("reader_pair_found_flag"))
        pair_punch_count = self._optional_float(row.get("pair_punch_count")) or 0
        return pair_punch_count >= 2 and self._clean_shift_value(row.get("best_shift_candidate")) is not None

    def _clean_shift_value(self, value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None

        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "unknown", "-"}:
            return None
        return text

    def _optional_float(self, value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None

        numeric_value = pd.to_numeric(value, errors="coerce")
        if pd.isna(numeric_value):
            return None
        return float(numeric_value)

    def _optional_int(self, value: Any) -> int | None:
        numeric_value = self._optional_float(value)
        if numeric_value is None:
            return None
        return int(numeric_value)

    def _json_safe_dict(self, value: dict[str, Any]) -> dict[str, Any]:
        return json.loads(pd.Series(value).to_json(date_format="iso"))


shift_prediction_service = ShiftPredictionService()
