# 📚 Pipeline de préparation du dataset FinBERT

Ce projet contient deux étapes principales pour transformer les rapports financiers bruts en tenseurs exploitables par un modèle FinBERT.

## ⚙️ Étape 1 – Préparation des données brutes

Script : prep_text_dataset.py
- Charge les fichiers text_us_YYYY.pkl (2005–2025).
- Concatène toutes les années en un seul DataFrame Polars.
- Filtre les rapports :
- uniquement les tickers S&P500
- uniquement les 10Q
- longueur ≤ 250 000 caractères
- rf et mgmt non vides
- garde uniquement les CIK les plus fréquents (mode)
- Ajoute les métadonnées du S&P500 (ticker, security).
- Vérifie la cohérence du dataset.
- Sauvegarde en Parquet (text_dataset.parquet).

👉 Exécution :

```bash
python src/text_processing/prep_text_dataset.py
```
## ⚙️ Étape 2 – Tokenization & Chunking

Script : prep_dataset.py
- Charge le dataset filtré depuis Hugging Face ou un fichier parquet local.
- Tokenize les textes avec FinBERT.
- Découpe chaque texte en chunks de taille fixe (par défaut 256 tokens × 10 chunks).
- Produit des tenseurs PyTorch (input_ids, attention_masks, labels).
- Sauvegarde en .pt (finbert_chunks.pt).

👉 Exécution :

```bash
python src/text_processing/prep_dataset.py \
    --repo_id "Arthurmaffre34/predataset" \
    --data_file "pre_dataset.parquet" \
    --max_length 256 \
    --n_chunks 10 \
    --save_path "finbert_chunks.pt"
```

# 📂 Gestion des chemins (datasets)

Nous utilisons **`pathlib`** pour définir des chemins **dynamiques et portables**, afin que le code fonctionne sur n’importe quelle machine sans modifier les chemins manuellement.

```python
from pathlib import Path

# Base du projet (2 niveaux au-dessus du script courant)
BASE_DIR = Path.cwd().parents[2]

# Répertoire datasets
DATASET_DIR = BASE_DIR / "src" / "text_processing" / "text_dataset"

# Fichiers utilisés dans le pipeline
DATASET_OUTPUT_FILE = DATASET_DIR / "text_dataset.parquet"
SP500_OUTPUT_FILE   = DATASET_DIR / "sp500.parquet"
OUT_FILTER_FILE     = DATASET_DIR / "out_filter_dataset.parquet"
PRE_DATASET_FILE    = DATASET_DIR / "pre_dataset.parquet"

if __name__ == "__main__":
    print("Base dir:", BASE_DIR)
    print("Dataset dir:", DATASET_DIR)
```
## Exemple d'utilisation

```python
import pandas as pd
from pathlib import Path

from config_paths import PRE_DATASET_FILE  # <- ton module avec les chemins

# Charger le dataset prétraité
df = pd.read_parquet(PRE_DATASET_FILE)
print("Shape:", df.shape)
print(df.head())
```


### ✅ Avantages
- **Évolutif** : pas de chemins absolus → portable partout.  
- **Centralisé** : tous les chemins importants au même endroit.  
- **Clair** : on sait exactement où sont stockés les datasets.  
