#importing libs
import polars as pl
import pandas as pd
import yfinance as yf
import warnings

#importing paths
from paths import DATASET_OUTPUT_FILE, PRE_DATASET_FILE

def load_text_dataset(path = DATASET_OUTPUT_FILE):
    dataframe = pl.read_parquet(DATASET_OUTPUT_FILE)
    dataframe = dataframe.to_pandas()

    dataframe["date"] = pd.to_datetime(dataframe["date"]) # ne pas toucher aux index
    return dataframe


def download_sp500_from_yahoo(dataframe, offset_days = 63):
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    tickers = dataframe["Symbol"].unique().tolist()

    min_date = dataframe["date"].min()
    max_date = dataframe["date"].max() + pd.DateOffset(days=offset_days)

    sp500_raw_dataframe = yf.download(
        tickers,
        start=min_date,
        end=max_date
    )['Close']

    dataframe.reset_index(inplace = True) # ne pas toucher aux index
    dataframe["date"] = pd.to_datetime(dataframe["date"])   # conversion en datetime
    dataframe = dataframe.set_index(["Symbol", "date"])                 # nouvel index
    return dataframe, sp500_raw_dataframe

def compute_forward_returns(data, offset_days = 63):
    data_returns = data.shift(-offset_days) / data - 1
    data_returns = data_returns.iloc[:-63]
    return data_returns

def apply_returns_on_dataframe(dataframe, data_returns):
    dates_dataframe = dataframe.index.get_level_values("date").unique()
    data_returns = data_returns.loc[data_returns.index.intersection(dates_dataframe)]

    data_returns = data_returns.reset_index().rename(columns={"index": "date"})

    data_returns = data_returns.melt(
        id_vars="date", var_name="Symbol", value_name="return"
    )

    data_returns = data_returns.set_index(["Symbol", "date"])

    dataframe = dataframe.join(data_returns, how="left")
    dataframe = dataframe[["mgmt", "rf", "return"]]

    return dataframe

def clean_dataframe(dataframe):
    dataframe[["mgmt", "rf", "return"]]
    dataframe = dataframe.dropna(subset=["return", "mgmt", "rf"])
    return dataframe

def export(dataframe):
    dataframe.to_parquet(PRE_DATASET_FILE)



def prepare_dataset():
    dataframe = load_text_dataset(path = DATASET_OUTPUT_FILE)

    dataframe, data = download_sp500_from_yahoo(dataframe, offset_days = 63)

    data_returns = compute_forward_returns(data)

    dataframe = apply_returns_on_dataframe(dataframe, data_returns)

    dataframe = clean_dataframe(dataframe)

    export(dataframe)

if __name__ == "__main__":
    prepare_dataset()