# Shift Prediction Web App

Local FastAPI web app for uploading transaction files and running the final V3 day-first reader production shift engine from `shift_engine_package_v3_day_first_reader`.

## Project Structure

```text
shift-prediction/
├── backend/
│   ├── config.py
│   ├── main.py
│   ├── schemas.py
│   └── services/
│       ├── comparison_service.py
│       ├── file_service.py
│       └── shift_service.py
├── frontend/
│   ├── comparison.html
│   ├── index.html
│   └── assets/
│       ├── app.js
│       ├── comparison.css
│       ├── comparison.js
│       └── styles.css
├── outputs/
├── shift_engine_package/
├── uploads/
├── requirements.txt
└── README.md
```

## Features

- Upload `.xls`, `.xlsx`, and `.csv` transaction files
- Validate required columns: `EmpCode`, `TransactionDateTime`, `ReaderNumber`
- Run the final V3 day-first reader engine from `shift_engine_package_v3_day_first_reader/inference.py`
- Show summary cards, searchable results table, and pagination
- Save output CSV files into `outputs/`
- Download generated predictions from the web UI
- Compare prediction outputs against an attendance benchmark on a separate comparison page

## Setup

1. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   ```

2. Activate it:

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run Locally

From the project root:

```bash
uvicorn backend.main:app --reload
```

Open the app at:

```text
http://127.0.0.1:8000
```

## API Endpoints

- `GET /api/health`
- `POST /api/predict`
- `GET /api/download/{file_name}`
- `GET /comparison`
- `POST /api/compare`
- `GET /api/download-comparison/{file_name}`

## Expected Input

Required columns:

- `EmpCode`
- `TransactionDateTime`
- `ReaderNumber`

`ReaderNumber = 1` is treated as Punch In and `ReaderNumber = 2` is treated as Punch Out.

Final V3 day-first reader rules:

- Punch In is the earliest same-day `ReaderNumber = 1`.
- Punch Out first uses the latest same-day `ReaderNumber = 2` after Punch In.
- If no same-day OUT exists, the engine falls back to latest OUT within 16 hours, then 24 hours, then 48 hours.
- If a same-day OUT exists, next-day punches after the selected OUT are ignored for scoring/manual-continuous logic.
- No same-day transaction is `ABSENT`.
- Sunday with same-day transaction is `WO_REVIEW`.
- Supported model classes are `PF`, `PGM`, `PS`, and `PT`.

Optional columns supported by the engine:

- `ReaderId`
- `ReasonCode`
- `TransactionCode`

## Usage

1. Open the website.
2. Upload a transaction file.
3. Click `Run Prediction`.
4. Review summary counts and detailed rows.
5. Download the generated CSV if needed.

## Comparison Page

The comparison workflow is separate from the transaction prediction page. Attendance files are treated only as benchmark / answer-key data and are not used by the shift engine.

1. Run prediction from the main page.
2. Click the top `Comparison` button, or open:

   ```text
   http://127.0.0.1:8000/comparison
   ```

3. Upload the raw transaction file, such as `pat_transaction_data.xls`.
4. Upload the attendance truth file, such as `pat_att_data.xls`.
5. Click `Run Prediction + Comparison`.
6. Review summary cards, search/filter comparison rows, and download the generated comparison CSV.

The comparison page runs the same shift engine on the uploaded raw transaction file, then matches generated prediction rows against attendance rows by employee id and date. The main KPI is predicted final shift vs normalized attendance `ShiftShortName`.

- Coverage reports attendance rows, prediction rows, matched rows, missing in prediction, missing in attendance, and coverage percentage.
- Attendance shift family normalization maps `PFW -> PF`, `PF -> PF`, `PSW -> PS`, `PS -> PS`, `PTW -> PT`, `PT -> PT`, `PG -> PGM`, `PGM -> PGM`, and `PGW -> PGM`.
- Shift answer-key accuracy is calculated only on matched rows where both sides are supported working shift families. WO, leave, absent, missing truth shift, missing prediction, missing attendance, and unsupported shifts such as `KFS`, `KSS`, `MN`, `MF`, `MS`, and `PD` are excluded.
- Review rows are reported separately as review matches or review mismatches instead of being mixed with hard predicted-shift errors.

## Notes

- Uploaded files are stored temporarily in `uploads/` and deleted after prediction completes.
- Prediction outputs are written to `outputs/`.
- Comparison outputs are written to `outputs/` with a `comparison-` filename prefix.
- Current active inference logic is `shift_engine_package_v3_day_first_reader/inference.py` and `shift_engine_package_v3_day_first_reader/artifacts/`. The old engine folders are kept as backup only.
