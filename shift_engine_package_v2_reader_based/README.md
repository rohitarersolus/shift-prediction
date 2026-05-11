# Shift Engine Package V2 - ReaderNumber Based

This package is the production website engine for shift prediction.

## Final logic

- ReaderNumber = 1 is treated as Punch In.
- ReaderNumber = 2 is treated as Punch Out.
- Earliest ReaderNumber 1 and latest ReaderNumber 2 are used to build valid shift pair features.
- If an employee has no same-day transaction, the row is marked as ABSENT.
- If Sunday has same-day transaction, it is marked as WO_REVIEW.
- LightGBM predicts only working shift classes: PF, PGM, PS, PT.

## Required input columns

- EmpCode
- TransactionDateTime
- ReaderNumber

## Optional input columns

- ReaderId
- ReasonCode
- TransactionCode
- PersonName

## Run locally

pip install -r requirements.txt
python app.py /path/to/pat_transaction_data.xls --output-dir outputs

Debug output:

python app.py /path/to/pat_transaction_data.xls --output-dir outputs --debug

## Python usage

from inference import run_inference
df = run_inference('pat_transaction_data.xls')
print(df.head())

## Output files

- prediction_clean.csv - user/business friendly output
- prediction_debug.csv - internal/debug output