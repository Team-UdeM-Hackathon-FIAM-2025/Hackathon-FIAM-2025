#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de préparation du dataset FinBERT avec chunking et sauvegarde en .pt
"""
import argparse
import torch
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset

def load_hf_dataset(repo_id: str, data_file: str) -> pd.DataFrame:
    """
    Charge un dataset Hugging Face et retourne un DataFrame Pandas.
    
    Args:
        repo_id (str): ID du repo Hugging Face.
        data_file (str): Nom du fichier (par ex. parquet ou csv).

    Returns:
        pd.DataFrame
    """
    dataset = load_dataset(repo_id, data_files=data_file)
    df = dataset["train"].to_pandas()
    print(f"✅ Dataset chargé depuis {repo_id}/{data_file} : {df.shape}")
    return df

def tokenize_and_chunk(texts, tokenizer, max_length=512, n_chunks=10, desc="Chunking"):
    """
    Tokenize une liste de textes et les divise en chunks de taille fixe.
    
    Args:
        texts (list[str]): Liste des textes à encoder.
        tokenizer: Tokenizer Hugging Face (AutoTokenizer).
        max_length (int): Taille max d’un chunk.
        n_chunks (int): Nombre max de chunks par texte.
        desc (str): Label pour la barre de progression.

    Returns:
        input_ids (torch.Tensor): Tensor [N, n_chunks, max_length]
        attention_mask (torch.Tensor): Tensor [N, n_chunks, max_length]
    """
    all_input_ids, all_attention = [], []

    for text in tqdm(texts, desc=desc):
        tokens = tokenizer.encode(text, add_special_tokens=True)

        # ⚠️ Retirer le CLS (1er) et SEP (dernier) pour éviter la duplication
        if tokens[0] == tokenizer.cls_token_id:
            tokens = tokens[1:]
        if tokens[-1] == tokenizer.sep_token_id:
            tokens = tokens[:-1]

        # Découpage en chunks de (max_length - 2)
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

        # Padding si moins de n_chunks
        while len(ids) < n_chunks:
            ids.append([tokenizer.pad_token_id] * max_length)
            masks.append([0] * max_length)

        all_input_ids.append(ids)
        all_attention.append(masks)

    return torch.tensor(all_input_ids), torch.tensor(all_attention)


def build_dataset(df, tokenizer_name="yiyanghkust/finbert-pretrain",
                  max_length=256, n_chunks=20):
    """
    Construit les tenseurs RF/MGMT à partir d’un DataFrame Pandas.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    rf_ids, rf_masks = tokenize_and_chunk(
        df["rf"].astype(str).tolist(),
        tokenizer,
        max_length=max_length,
        n_chunks=n_chunks,
        desc="Tokenizing RF"
    )

    mgmt_ids, mgmt_masks = tokenize_and_chunk(
        df["mgmt"].astype(str).tolist(),
        tokenizer,
        max_length=max_length,
        n_chunks=n_chunks,
        desc="Tokenizing MGMT"
    )

    labels = torch.tensor(df["return"].values, dtype=torch.float)

    return {
        "rf_input_ids": rf_ids,
        "rf_attention_mask": rf_masks,
        "mgmt_input_ids": mgmt_ids,
        "mgmt_attention_mask": mgmt_masks,
        "labels": labels
    }


def save_dataset(dataset_dict, save_path="finbert_chunks.pt"):
    """
    Sauvegarde le dataset prétraité en .pt
    """
    torch.save(dataset_dict, save_path)
    print(f"💾 Dataset sauvegardé dans {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Préparer le dataset FinBERT depuis Hugging Face.")

    parser.add_argument(
        "--repo_id", type=str, required=False,
        default="Arthurmaffre34/predataset",
        help="ID du repo Hugging Face (default: Arthurmaffre34/predataset)"
    )

    parser.add_argument(
        "--data_file", type=str, required=False,
        default="pre_dataset.parquet",
        help="Nom du fichier du dataset (default: pre_dataset.parquet)"
    )

    parser.add_argument(
        "--max_length", type=int, required=False,
        default=256,
        help="Taille max d’un chunk (default: 256)"
    )

    parser.add_argument(
        "--n_chunks", type=int, required=False,
        default=10,
        help="Nombre max de chunks par texte (default: 20)"
    )

    parser.add_argument(
        "--save_path", type=str, required=False,
        default="finbert_chunks.pt",
        help="Chemin du fichier de sortie (default: finbert_chunks.pt)"
    )

    args = parser.parse_args()

    # Charger dataset brut depuis HF
    df = load_hf_dataset(args.repo_id, args.data_file)

    # Construire dataset tokenizé/chunké
    dataset_dict = build_dataset(
        df,
        max_length=args.max_length,
        n_chunks=args.n_chunks
    )

    # Sauvegarder
    save_dataset(dataset_dict, args.save_path)


if __name__ == "__main__":
    main()