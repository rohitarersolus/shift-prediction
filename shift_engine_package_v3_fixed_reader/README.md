# Shift Engine Package V3 - Fixed Reader Pair

This package is the production website engine for shift prediction.

## Final logic

- ReaderNumber = 1 is treated as Punch In.
- ReaderNumber = 2 is treated as Punch Out.
- Punch In is earliest same-day ReaderNumber 1.
- Punch Out is latest ReaderNumber 2 after Punch In within the 36-hour window.
- The same fixed punch pair is scored against all shift classes.
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