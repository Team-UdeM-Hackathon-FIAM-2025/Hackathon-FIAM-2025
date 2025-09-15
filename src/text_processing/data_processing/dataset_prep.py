from datasets import load_dataset
import tqdm

ds = load_dataset("Gill-Hack-25-UdeM/raw_text_hack_2025", data_files="dataset.parquet", split="train")
print(ds)