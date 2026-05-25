from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Union

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


class PredictionFilterOptions(BaseModel):
    statuses: list[str]
    shifts: list[str]
    mismatch_states: list[str]


class PredictionPageResponse(BaseModel):
    output_file_name: str
    columns: list[str]
    page: int
    page_size: int
    total_pages: int
    total_rows: int
    filtered_row_count: int
    filters: PredictionFilterOptions
    data: list[dict[str, Any]]


class PredictResponse(BaseModel):
    file_name: str
    output_file_name: str
    generated_at: datetime
    summary: PredictionSummary
    row_count: int
    columns: list[str]
    page: int
    page_size: int
    total_pages: int
    filtered_row_count: int
    filters: PredictionFilterOptions
    download_url: str
    debug_download_url: Optional[str] = None
    data: list[dict[str, Any]]


class ManualPredictRequest(BaseModel):
    employee_id: str
    date: str
    punch_in: str
    punch_out: str
    extra_punches: Optional[Union[list[str], str]] = None
    note: Optional[str] = None


class ManualPredictResponse(BaseModel):
    employee_id: str
    date: str
    model_predicted_shift: Optional[str]
    model_confidence: Optional[float]
    historical_regular_shift: str
    history_confidence: Optional[float] = None
    history_consistency: Optional[str] = None
    history_support: Optional[int] = None
    history_source: Optional[str] = None
    final_recommended_shift: str
    status: str
    message: str
    raw_transactions_used: list[str]
    raw_transactions_text: str
    note: Optional[str] = None
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
    debug_download_url: Optional[str] = None
    data: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    detail: str
    detected_columns: Optional[list[str]] = None
    supported_aliases: Optional[dict[str, list[str]]] = None
