# 🚀 Hackathon Model V2 – Quick Start  

Ce repo permet de tester **notre modèle ONNX** directement depuis Hugging Face Hub.  
👉 Pas besoin de télécharger le modèle manuellement : tout est fait **automatiquement** par le script.  

---

## ▶️ Lancer le test automatique
Exécute simplement :  

```bash
python inference.py
```

👉 Le script va :  
1. **Télécharger automatiquement** le modèle ONNX depuis Hugging Face.  
2. Générer un petit fichier **`test_input.csv`** avec des exemples.  
3. Faire tourner l’inférence.  
4. Sauvegarder les résultats dans **`output.csv`** avec une nouvelle colonne `pred_return`.  

---

## 📊 Exemple de sortie
Après exécution, tu verras un fichier `output.csv` comme ça :  

| date       | cik   | rf                                    | mgmt                                   | pred_return |
|------------|-------|---------------------------------------|----------------------------------------|-------------|
| 2005-01-06 | 23217 | Risk factor indicates potential …     | Management discussion shows growth      | 0.0342      |
| 2005-01-07 | 40704 | Company faces litigation risk …       | Management expects steady expansion     | -0.0121     |

---

## 🛠 Utilisation sur ton propre dataset
Prépare un **CSV** avec au minimum les colonnes suivantes :  
- `date`  
- `cik`  
- `rf` (Risk Factor text)  
- `mgmt` (Management Discussion text)  

Puis lance :  
```bash
python inference.py --input_csv mon_dataset.csv --output_csv resultats.csv
```

Le script ajoutera automatiquement la colonne **`pred_return`** avec les prédictions.  

---

✅ **En résumé :**  
- Tu lances `python inference.py` → ça marche direct.  
- Pas besoin de télécharger le modèle → c’est géré automatiquement.  
- Les résultats apparaissent dans `output.csv`.  
