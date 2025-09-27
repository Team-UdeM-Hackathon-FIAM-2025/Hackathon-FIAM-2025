import pandas as pd
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from tqdm import tqdm
from huggingface_hub import hf_hub_download



def download_model(
    repo_id: str = "Arthurmaffre34/Hackathon-mod-v2",
    filename: str = "best_model_epoch1.onnx",
    tokenizer_name: str = "yiyanghkust/finbert-pretrain"
):
    """
    Télécharge et charge un modèle ONNX depuis Hugging Face.

    Args:
        repo_id (str): ID du repo Hugging Face (par défaut: Arthurmaffre34/Hackathon-mod-v2).
        filename (str): Nom du fichier modèle dans le repo.
        tokenizer_name (str): Nom du tokenizer Hugging Face à charger.

    Returns:
        session (onnxruntime.InferenceSession): session ONNX prête à l'inférence.
        tokenizer (AutoTokenizer): tokenizer associé.
    """
    # Télécharger le modèle depuis Hugging Face Hub
    onnx_model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model"
    )

    # Charger le modèle ONNX
    session = ort.InferenceSession(onnx_model_path)

    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    print(f"✅ Modèle ONNX chargé depuis {repo_id}/{filename}")
    return session, tokenizer

def tokenize_and_chunk_single(text, tokenizer, max_length=256, n_chunks=10):
    """Tokenize + chunk un texte en [1, n_chunks, max_length]."""
    if not isinstance(text, str):
        text = str(text)

    tokens = tokenizer.encode(text, add_special_tokens=True)

    # Retirer CLS et SEP pour éviter duplication
    if tokens and tokens[0] == tokenizer.cls_token_id:
        tokens = tokens[1:]
    if tokens and tokens[-1] == tokenizer.sep_token_id:
        tokens = tokens[:-1]

    chunks = [
        [tokenizer.cls_token_id] + tokens[i:i+(max_length-2)] + [tokenizer.sep_token_id]
        for i in range(0, len(tokens), max_length-2)
    ]

    ids, masks = [], []
    for chunk in chunks[:n_chunks]:
        attn = [1] * len(chunk)
        if len(chunk) < max_length:
            pad_len = max_length - len(chunk)
            chunk = chunk + [tokenizer.pad_token_id] * pad_len
            attn = attn + [0] * pad_len
        ids.append(chunk)
        masks.append(attn)

    while len(ids) < n_chunks:
        ids.append([tokenizer.pad_token_id] * max_length)
        masks.append([0] * max_length)

    return np.array([ids], dtype=np.int64), np.array([masks], dtype=np.int64)


def run_inference_on_row(rf_text, mgmt_text):
    """Applique l'inférence ONNX sur une ligne (rf, mgmt)."""
    rf_ids, rf_masks     = tokenize_and_chunk_single(rf_text, tokenizer)
    mgmt_ids, mgmt_masks = tokenize_and_chunk_single(mgmt_text, tokenizer)

    inputs = {
        "rf_ids": rf_ids,
        "rf_masks": rf_masks,
        "mgmt_ids": mgmt_ids,
        "mgmt_masks": mgmt_masks,
    }
    outputs = session.run(None, inputs)
    return float(outputs[0].squeeze())


def run_inference_on_csv(input_csv, output_csv="predictions.csv"):
    """Charge un CSV, applique inférence, sauvegarde avec pred_return."""
    df = pd.read_csv(input_csv)

    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        pred = run_inference_on_row(row["rf"], row["mgmt"])
        preds.append(pred)

    df["pred_return"] = preds
    df.to_csv(output_csv, index=False)
    print(f"✅ Résultats sauvegardés dans {output_csv}")
    return df


import argparse

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Batch inference with ONNX model")
    parser.add_argument("--input_csv", type=str, help="Path to input CSV", required=False)
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Path to save output CSV")
    args = parser.parse_args()

    session, tokenizer = download_model()

    if args.input_csv: # Mode utilisateur : utilise un fichier CSV fourni
        input_csv = args.input_csv
        
    else: # Mode démo : crée un dataset factice
        test_data = [
            {"date": "2005-01-06", "cik": 23217, "rf": "Risk factor indicates potential losses in overseas operations", "mgmt": "Management discussion shows strong revenue growth"},
            {"date": "2005-01-07", "cik": 40704, "rf": "Company faces litigation risk from pending lawsuits", "mgmt": "Management expects steady market expansion"},
            {"date": "2005-01-08", "cik": 764478, "rf": "Supply chain disruptions may affect Q2 production", "mgmt": "Management highlights efficiency improvements"},
        ]

        df_test = pd.DataFrame(test_data)
        df_test.to_csv("test_input.csv", index=False)
        print("✅ Fichier test_input.csv créé")
        input_csv = "test_input.csv"

    df_pred = run_inference_on_csv(input_csv, args.output_csv)
    print("\n✅ Inference terminée ! Résultats dans :", args.output_csv)
    print(df_pred.head())
    print(df_pred.shape)