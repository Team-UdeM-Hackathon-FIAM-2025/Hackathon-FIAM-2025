#importing libs
import os
import pickle as pkl
import polars as pl
import gc
import warnings
import gc

#importing paths
from paths import DATASET_DIR, DATASET_OUTPUT_FILE, SP500_OUTPUT_FILE

def load_datasets_polars( # Step 1 loading dataset files
    path: str = DATASET_DIR,
    start_date: int = 2005,
    end_date: int = 2025,
    ignore_warnings: bool = True
) -> pl.DataFrame:
    
    
    if ignore_warnings == True: # Suppress DeprecationWarnings during dataset loading
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset folder not found: {path}")

    df_all = None  # empty df init
    total_rows = 0  # Cumulative row counter

    for year in range(start_date, end_date + 1):
        file_path = os.path.join(path, f"text_us_{year}.pkl")
        try:
            with open(file_path, "rb") as f:
                df = pkl.load(f)  # Pandas DataFrame
            df["year"] = year
            df_polars = pl.from_pandas(df)
            del df
            gc.collect()

            # Row counter
            total_rows += df_polars.height
            print(f"[OK] Loaded text_us_{year}.pkl ({df_polars.height} lignes, cumul={total_rows})")

            if df_all is None:
                df_all = df_polars
            else:
                df_all = pl.concat([df_all, df_polars], how="vertical")

        except FileNotFoundError:
            raise FileNotFoundError(f"Missing dataset for year {year}: {file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading {file_path}: {e}")

    df_all = df_all.with_columns([
        pl.col("date").str.strptime(pl.Date, "%Y%m%d", strict=False)
    ])

    return df_all


# Filter functions
def apply_sp_500_filter(dataframe, path_filter = SP500_OUTPUT_FILE):
    df_sp500 = pl.read_parquet(path_filter)

    dataframe = dataframe.join(df_sp500.select("cik"), on="cik", how="inner") #filtrage des tickers du sp500
    return dataframe

def filter_file_type(dataframe): #filtrage des rapports seulement 10Q
    dataframe = dataframe.filter(pl.col("file_type") == "10Q")
    return dataframe

def filter_length(dataframe, max_length = 250_000):
    dataframe = dataframe.filter(
        (pl.col("rf").str.len_chars() <= max_length) &
        (pl.col("mgmt").str.len_chars() <= max_length)
    )
    return dataframe

def filter_df_mgmt_full(dataframe):
    dataframe = dataframe.filter(
        (pl.col("rf").str.strip_chars() != "") &
        (pl.col("mgmt").str.strip_chars() != "")
    )
    return dataframe

def filter_keep_mode_cik(dataframe: pl.DataFrame) -> pl.DataFrame:
    # Compter le nombre d'occurrences par CIK
    counts = dataframe.group_by("cik").agg(pl.count().alias("count"))
    # Trouver la valeur du mode (la fréquence max la plus fréquente)
    mode_count = counts["count"].mode()[0]
    # Garder uniquement les CIK qui apparaissent ce nombre de fois
    cik_mode = counts.filter(pl.col("count") == mode_count)["cik"]
    return dataframe.filter(pl.col("cik").is_in(cik_mode))


def add_sp500_info(dataframe, path_filter = SP500_OUTPUT_FILE):
    df_sp500 = pl.read_parquet(path_filter)

    dataframe = dataframe.join(
        df_sp500.select(["cik", "Symbol", "Security"]),
        on="cik",
        how="left"
    )
    return dataframe


# utils functions
def verify_filters(dataframe, path_filter = SP500_OUTPUT_FILE): #verification function
    # 1 : Vérifier qu'aucun rf ou mgmt n'est vide
    assert not ((dataframe["rf"].str.strip_chars() == "").any()), "⚠️ Certains 'rf' sont vides"
    assert not ((dataframe["mgmt"].str.strip_chars() == "").any()), "⚠️ Certains 'mgmt' sont vides"

    # 2 : Vérifier qu'il n'y a que les formulaires 10Q
    assert (dataframe["file_type"] == "10Q").all(), "⚠️ Certains file_type sont différents de 10Q"

    # 3 : Vérifier que les longueurs sont OK
    assert not (dataframe["rf"].str.len_chars() > 250_000).any(), "⚠️ Certains 'rf' dépassent 250k caractères"
    assert not (dataframe["mgmt"].str.len_chars() > 250_000).any(), "⚠️ Certains 'mgmt' dépassent 250k caractères"

    # 4 : Vérification du nombre de CIK distincts
    unique_cik_count = dataframe.select(pl.col("cik").n_unique()).item()
    assert unique_cik_count > 0, "⚠️ Aucun CIK trouvé dans df_mode_cik"

    # 5 : Vérifier que tous les CIK sont dans df_sp500
    df_sp500 = pl.read_parquet(path_filter)
    missing_cik = dataframe.join(df_sp500.select("cik"), on="cik", how="anti")
    assert missing_cik.shape[0] == 0, f"⚠️ {missing_cik.shape[0]} CIK manquants dans S&P500"

def save_dataset(dataframe, path_output = DATASET_OUTPUT_FILE):
    dataframe.write_parquet(DATASET_OUTPUT_FILE)




def prep_dataset():
    # parameters
    ignore_warnings = True

    df = load_datasets_polars(path=DATASET_DIR, start_date=2005, #executing step 1
                        end_date=2025, ignore_warnings = ignore_warnings)
    print("Loaded datasets: step 1 completed")


    df = apply_sp_500_filter(dataframe = df, path_filter = SP500_OUTPUT_FILE) #executing step 2
    print("Applied sp_500 filter: step 2.1 completed")

    df = filter_file_type(dataframe = df)
    print("Applied 10KSB filter: step 2.2 completed")

    df = filter_length(dataframe = df)
    print("Applied length filter: step 2.3 completed")

    df = filter_df_mgmt_full(dataframe = df)
    print("Applied df_mgmt_full filter: step 2.4 completed")

    df = filter_keep_mode_cik(dataframe = df) #only keep entire sequences
    print("Applied keep_mode_cik filter: step 2.5 completed")


    df = add_sp500_info(dataframe = df, path_filter = SP500_OUTPUT_FILE)
    print("Add sp500 info to dataset: step 3 completed")

    # utils functions
    print("Starting verification")
    verify_filters(dataframe = df, path_filter = SP500_OUTPUT_FILE)
    print("✅ Toutes les vérifications sont passées avec succès !")

    save_dataset(dataframe = df, path_output = DATASET_OUTPUT_FILE)
    print(f"✅ Dataset successfully saved at: {DATASET_OUTPUT_FILE}")





if __name__ == "__main__":
    prep_dataset()