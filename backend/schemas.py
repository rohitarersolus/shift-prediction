from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    engine_version: str = "v3_day_first_reader"
    model_classes: list[str] = ["PF", "PGM", "PS", "PT"]
    reader_in_number: int = 1
    reader_out_number: int = 2
    punch_in_rule: str = "earliest same-day ReaderNumber 1"
    punch_out_rule: str = "same-day latest ReaderNumber 2 first, fallback 16h/24h/48h"


class PredictionSummary(BaseModel):
    total_rows: int
    shift_predicted: int = Field(alias="SHIFT_PREDICTED")
    shift_review: int = Field(alias="SHIFT_REVIEW")
    withheld: int = Field(alias="WITHHELD")
    absent: int = Field(alias="ABSENT")
    wo: int = Field(alias="WO")
    wo_review: int = Field(alias="WO_REVIEW")

    model_config = {"populate_by_name": True}


class PredictResponse(BaseModel):
    file_name: str
    output_file_name: str
    generated_at: datetime
    summary: PredictionSummary
    row_count: int
    columns: list[str]
    download_url: str
    debug_download_url: str | None = None
    data: list[dict[str, Any]]


class ManualPredictRequest(BaseModel):
    employee_id: str
    date: str
    punch_in: str
    punch_out: str
    extra_punches: list[str] | str | None = None
    note: str | None = None


class ManualPredictResponse(BaseModel):
    employee_id: str
    date: str
    model_predicted_shift: str | None
    model_confidence: float | None
    historical_regular_shift: str
    history_confidence: float | None = None
    history_consistency: str | None = None
    history_support: int | None = None
    history_source: str | None = None
    final_recommended_shift: str
    status: str
    message: str
    raw_transactions_used: list[str]
    raw_transactions_text: str
    note: str | None = None
    engine_row: dict[str, Any]


class ComparisonSummary(BaseModel):
    attendance_total_rows: int
    prediction_total_rows: int
    matched_rows: int
    missing_in_prediction: int
    missing_in_attendance: int
    coverage_percent: float
    comparable_working_shift_rows: int
    comparable_shift_rows: int = 0
    predicted_only_shift_accuracy_percent: float
    assigned_shift_accuracy_percent: float
    exact_shift_matches: int
    exact_shift_mismatches: int
    review_shift_matches: int
    review_shift_mismatches: int
    non_working_excluded: int
    shift_mismatches: int


class CompareResponse(BaseModel):
    output_file_name: str
    generated_at: datetime
    summary: ComparisonSummary
    row_count: int
    columns: list[str]
    download_url: str
    debug_download_url: str | None = None
    data: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    detail: str
    detected_columns: list[str] | None = None
    supported_aliases: dict[str, list[str]] | None = None
