# Installation — pipeline d'entraînement

Python **3.11**. Le projet a été développé sous `uv` ; les deux voies fonctionnent.

## uv (voie d'origine, recommandée)

```bash
cd training
uv sync                 # résout depuis pyproject.toml + uv.lock (pins exacts)
```

## pip

```bash
cd training
python -m venv .venv && . .venv/bin/activate     # Windows : .venv\Scripts\activate
pip install -r ../requirements.txt               # 7 dépendances directes, pinnées
# ou, pour l'environnement complet au pin près :
pip install -r ../requirements-lock.txt
```

## Variables d'environnement

| Variable | Rôle |
|---|---|
| `TINKER_API_KEY` | clé API Tinker (backend d'entraînement) |
| `TINKER_API_KEY_FILE` | *(alternative)* chemin d'un fichier contenant la clé |
| `KAGGLE_API_TOKEN` | requis seulement par `upload_adapter.py` (via `env.json`) |
| `EQSYM_PROBLEMS_DIR` | dossier `problems/` pour `symsolve/eqsym_re_round2.py` |
| `SUBMISSION_DIR` | dossier de sortie des `submission*.zip` (défaut : `training/submission/`) |

## Exécution

```bash
uv run python3 reasoning.py        # 1. CoT par famille via solveurs/reasoners
uv run python3 augmentation.py     # 2. augmentation synthétique (cipher, bit)
uv run python3 corpus.py           # 3. corpus SFT (trace gardée ssi verify(gold, boxed))
uv run python3 train_sft.py        # 4. LoRA rank-32 SFT — le run 0.85
uv run python3 convert_to_kaggle.py   # 5. Tinker -> format Kaggle (un-fuse MoE, QR+SVD Mamba)
uv run python3 package_for_kaggle.py  # 6. submission.zip
```

Hyperparamètres du run 0.85 (`Cfg` dans `train_sft.py`) : `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`,
LoRA rank 32, `max_length` 8192, batch 64, LR linéaire décroissante, `train_mlp/attn/unembed`,
batches stratifiés par catégorie.
