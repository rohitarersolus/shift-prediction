# Shift Engine Package

## What this package does
- Loads the trained LightGBM model
- Builds pair-based transaction features from raw transaction data
- Predicts working shifts
- Applies Sunday business rule:
  - Sunday with transaction -> WO_REVIEW
- Returns final business-friendly output

## Important
This package predicts only from transaction data.
It does not infer leave, absent, or public holiday from no-transaction days.

## Run locally
    pip install -r requirements.txt
    uvicorn app:app --reload

## API
POST `/predict` with a CSV/XLS/XLSX transaction file.

Required columns:
- EmpCode
- TransactionDateTime

Optional columns:
- ReaderId
- ReasonCode
- TransactionCode
