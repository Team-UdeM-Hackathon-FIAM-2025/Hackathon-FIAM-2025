# Team UdeM Hackathon FIAM 2025 🚀

This repository hosts the codebase and documentation for our participation in the **FIAM Asset Management Hackathon 2025** as **Team Université de Montréal**.  
Our objective is to design, implement, and present a **full pipeline for portfolio optimization and financial modeling**, from **data collection** to **evaluation and visualization**.

---

## 📌 Purpose of this Repository

- **Centralize all code and documentation** for the hackathon project.  
- **Enable collaborative development** using professional GitHub workflows.  
- **Ensure reproducibility and transparency** of our work (every commit and branch is tracked).  
- **Protect Intellectual Property** (Apache 2.0 license).  

This repository is structured as if it were a **professional software project**, to train and showcase best practices.

---

## 📦 Installation (requirements.txt)

Make sure you have Python 3.12+ installed. Then install all dependencies with:

```bash
pip install -r requirements.txt
```

If you are working in a clean virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

This will install all packages needed for:
- Data processing (pandas, polars, numpy)
- Machine learning & NLP (torch, transformers, onnxruntime)
- Visualization (matplotlib, plotly, seaborn)
- Utility (tqdm, dotenv, huggingface_hub)

---

## 🌱 GitHub Workflow (Professional Standard)

We follow a **Feature Branch Workflow**, with **Pull Requests (PRs)** and **Code Reviews**.

### Branching Strategy
- `main` → always stable, production-ready code.  
- `dev` → integration branch (staging before merging to `main`).  
- `feature/*` → one branch per feature, e.g. `feature/data-loader`, `feature/backtest-engine`.  
- `hotfix/*` → urgent bug fixes.  

✅ **No direct commits to `main` are allowed.**

### Workflow Steps
1. **Create a branch for your feature**  
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/my-feature
   ```
2. **Work locally, commit often**  
   ```bash
   git add .
   git commit -m "Add data preprocessing pipeline"
   ```
3. **Push your branch**  
   ```bash
   git push origin feature/my-feature
   ```
4. **Open a Pull Request (PR) → dev**  
   - Assign a reviewer.  
   - Describe clearly what the PR does.  
   - One teammate must approve before merging.  

5. **Merge PR → dev**  
   - After testing, `dev` can be merged into `main` before submission/demo.

---

## 🔧 Useful Git Commands

Clone repo:
```bash
git clone https://github.com/Team-UdeM-Hackathon-FIAM-2025/<repo-name>.git
cd <repo-name>
```

Check branches:
```bash
git branch -a
```

Update local dev branch:
```bash
git checkout dev
git pull origin dev
```

Create new feature branch:
```bash
git checkout -b feature/new-model
```

Push branch:
```bash
git push origin feature/new-model
```

Sync fork (if needed):
```bash
git fetch origin
git merge origin/dev
```

---

## 📂 Repository Structure

```
/data/           -> Datasets (not pushed if too large, use .gitignore)
/notebooks/      -> Exploratory Jupyter notebooks
/src/            -> Core source code (modules, pipeline)
/tests/          -> Unit tests
/docs/           -> Documentation, slides
README.md        -> This file
```

---

## 🧑‍🤝‍🧑 Collaboration Rules

- **One branch per feature.**  
- **Always open a PR** (no direct push to `main`).  
- **Respect code reviews** (at least one approval required).  
- **Document your code** (docstrings, comments).  
- **Use meaningful commit messages** (`feat: add loader`, `fix: bug in preprocessing`, `refactor: clean model pipeline`).  

---

## 📜 License

This project is licensed under **Apache 2.0** – free to use, modify, and share with proper attribution.  
