from pathlib import Path
import tempfile

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from inference import ShiftEngine

app = FastAPI(title="Shift Classification API")
engine = ShiftEngine(artifacts_dir=str(Path(__file__).parent / "artifacts"))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".csv"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result_df = engine.predict_from_file(tmp_path)
        return JSONResponse({
            "rows": len(result_df),
            "data": result_df.to_dict(orient="records")
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
