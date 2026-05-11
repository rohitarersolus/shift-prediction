
import argparse
from pathlib import Path

from inference import run_inference, run_inference_debug


def main():
    parser = argparse.ArgumentParser(description="Shift Engine V2 Reader-Based Inference")
    parser.add_argument("input_file", help="Path to transaction file: .xls, .xlsx, or .csv")
    parser.add_argument("--output-dir", default="outputs", help="Output folder")
    parser.add_argument("--debug", action="store_true", help="Export debug output instead of clean output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        df = run_inference_debug(args.input_file, output_dir=output_dir)
        out_path = output_dir / "prediction_debug.csv"
    else:
        df = run_inference(args.input_file, output_dir=output_dir)
        out_path = output_dir / "prediction_clean.csv"

    print("Rows:", len(df))
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
