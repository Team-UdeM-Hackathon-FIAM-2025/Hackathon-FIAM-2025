from pathlib import Path

# Dossier du fichier courant (paths.py)
FILE_DIR = Path(__file__).resolve().parent

# Racine du projet = 3 niveaux au-dessus (adapter si besoin)
BASE_DIR = FILE_DIR.parents[2]

# Datasets
DATASET_DIR = BASE_DIR / "src" / "text_processing" / "text_dataset"
DATASET_OUTPUT_FILE = DATASET_DIR / "text_dataset.parquet"
SP500_OUTPUT_FILE   = DATASET_DIR / "sp500.parquet"
OUT_FILTER_FILE     = DATASET_DIR / "out_filter_dataset.parquet"
PRE_DATASET_FILE    = DATASET_DIR / "pre_dataset.parquet"

if __name__ == "__main__":
    print("File dir:", FILE_DIR)
    print("Base dir:", BASE_DIR)
    print("Dataset dir:", DATASET_DIR)
    print("Dataset output file:", DATASET_OUTPUT_FILE)
    print("SP500 output file:", SP500_OUTPUT_FILE)