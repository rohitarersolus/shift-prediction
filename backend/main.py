import logging
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from backend.config import ensure_runtime_directories, settings
from backend.schemas import (
    CompareResponse,
    ErrorResponse,
    HealthResponse,
    ManualPredictRequest,
    ManualPredictResponse,
    PredictResponse,
)
from backend.services.file_service import AppError, delete_file, save_upload_file
from backend.services.comparison_service import comparison_service
from backend.services.shift_service import shift_prediction_service


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shift_prediction")

app = FastAPI(
    title="Shift Prediction Web App",
    version="1.0.0",
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event() -> None:
    ensure_runtime_directories()


app.mount("/assets", StaticFiles(directory=settings.frontend_assets_dir), name="assets")


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.response_body)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error during request processing", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred while processing the request."},
    )


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(settings.frontend_dir / "index.html")


@app.head("/", include_in_schema=False)
async def index_head() -> JSONResponse:
    return JSONResponse(content={})


@app.get("/comparison", include_in_schema=False)
async def comparison() -> FileResponse:
    return FileResponse(settings.frontend_dir / "comparison.html")


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        engine_version=settings.engine_version,
        model_classes=shift_prediction_service.model_classes,
        reader_in_number=1,
        reader_out_number=2,
        punch_in_rule="earliest same-day ReaderNumber 1",
        punch_out_rule="same-day latest ReaderNumber 2 first, fallback 16h/24h/48h",
    )


@app.post("/api/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    saved_upload = await save_upload_file(file)

    try:
        prediction_result = await run_in_threadpool(
            shift_prediction_service.predict,
            file_path=saved_upload.path,
            original_name=saved_upload.original_name,
            use_cache=True,
        )
    finally:
        delete_file(saved_upload.path)

    return PredictResponse(
        file_name=prediction_result.file_name,
        output_file_name=prediction_result.output_file_name,
        generated_at=prediction_result.generated_at,
        summary=prediction_result.summary,
        row_count=prediction_result.summary["total_rows"],
        columns=prediction_result.columns,
        download_url=f"/api/download/{prediction_result.clean_output_file_name}",
        debug_download_url=f"/api/download/{prediction_result.output_file_name}",
        data=prediction_result.records,
    )


@app.post("/api/manual-predict", response_model=ManualPredictResponse)
async def manual_predict(payload: ManualPredictRequest) -> ManualPredictResponse:
    result = await run_in_threadpool(shift_prediction_service.manual_predict, payload.model_dump())
    return ManualPredictResponse(**result)


@app.get("/api/download/{file_name}", include_in_schema=False)
async def download_output(file_name: str) -> FileResponse:
    candidate_path = (settings.outputs_dir / Path(file_name).name).resolve()
    outputs_dir = settings.outputs_dir.resolve()

    if outputs_dir not in candidate_path.parents or not candidate_path.exists():
        raise AppError("Requested output file was not found.", status_code=404)

    return FileResponse(candidate_path, media_type="text/csv", filename=candidate_path.name)


@app.post("/api/compare", response_model=CompareResponse)
async def compare(
    transaction_file: UploadFile = File(...),
    attendance_file: UploadFile = File(...),
) -> CompareResponse:
    total_started = perf_counter()
    timings: dict[str, float] = {}

    started = perf_counter()
    saved_transaction = await save_upload_file(transaction_file)
    saved_attendance = await save_upload_file(attendance_file)
    timings["upload_save"] = perf_counter() - started

    try:
        started = perf_counter()
        prediction_result = await run_in_threadpool(
            shift_prediction_service.predict_for_comparison,
            file_path=saved_transaction.path,
            original_name=saved_transaction.original_name,
        )
        timings["prediction_or_cache"] = perf_counter() - started

        started = perf_counter()
        comparison_result = await run_in_threadpool(
            comparison_service.compare,
            attendance_path=saved_attendance.path,
            attendance_name=saved_attendance.original_name,
            prediction_path=prediction_result.output_path,
            prediction_name=prediction_result.output_file_name,
        )
        timings["comparison_only"] = perf_counter() - started
    finally:
        delete_file(saved_transaction.path)
        delete_file(saved_attendance.path)

    timings["total"] = perf_counter() - total_started
    logger.info(
        "Compare endpoint timing summary: upload_save=%.3fs prediction_or_cache=%.3fs comparison_only=%.3fs total=%.3fs prediction_output=%s",
        timings.get("upload_save", 0.0),
        timings.get("prediction_or_cache", 0.0),
        timings.get("comparison_only", 0.0),
        timings.get("total", 0.0),
        prediction_result.output_file_name,
    )

    return CompareResponse(
        output_file_name=comparison_result.output_file_name,
        generated_at=comparison_result.generated_at,
        summary=comparison_result.summary,
        row_count=comparison_result.summary["matched_rows"],
        columns=comparison_result.columns,
        download_url=f"/api/download-comparison/{comparison_result.clean_output_file_name}",
        debug_download_url=f"/api/download-comparison/{comparison_result.output_file_name}",
        data=comparison_result.records,
    )


@app.get("/api/download-comparison/{file_name}", include_in_schema=False)
async def download_comparison(file_name: str) -> FileResponse:
    candidate_path = (settings.outputs_dir / Path(file_name).name).resolve()
    outputs_dir = settings.outputs_dir.resolve()

    if (
        outputs_dir not in candidate_path.parents
        or not candidate_path.exists()
        or not candidate_path.name.startswith("comparison-")
    ):
        raise AppError("Requested comparison output file was not found.", status_code=404)

    return FileResponse(candidate_path, media_type="text/csv", filename=candidate_path.name)
