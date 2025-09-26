from pathlib import Path


BASE_DIR = Path.cwd().parents[2]
DATASET_DIR = BASE_DIR / "src" / "text_processing" / "text_dataset"
DATASET_OUTPUT_FILE = DATASET_DIR / "text_dataset.parquet"
SP500_OUTPUT_FILE = DATASET_DIR / "sp500.parquet"
OUT_FILTER_FILE = DATASET_DIR / "out_filter_dataset.parquet"
PRE_DATASET_FILE = DATASET_DIR / "pre_dataset.parquet"


if __name__ == "__main__":
    print("Base dir:", BASE_DIR)
    print("Dataset dir:", DATASET_DIR)
    print("Dataset output file:", DATASET_OUTPUT_FILE)
    print("SP500 output file:", SP500_OUTPUT_FILE)