# NVIDIA Nemotron Model Reasoning Challenge — livrables

Lot d'artefacts pour la soumission **« Reverse-engineering the ceiling »**
(track *Best Fine-tuning Method* / Open Contribution Award), compétition
`nvidia-nemotron-model-reasoning-challenge`.

Score leaderboard : **0.85** (LoRA rank-32 sur `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`,
pipeline entièrement reconstruit et contrôlé).

---

## Contenu

| Fichier / dossier | Quoi |
|---|---|
| `WRITEUP.md` | **Le writeup soumis à la compétition**, tel que publié. Titre : *Reverse-engineering the ceiling: a controlled 0.85 and a reproducible frontier map*. |
| `notebook/kaggle_writeup_notebook.ipynb` | **Le notebook Kaggle public**, tel qu'exécuté (vert) sur l'infra Kaggle. CPU, sans internet, rejoue les mesures du writeup sur `train.csv`. |
| `notebook/kernel-metadata.json` | Config de push du notebook (donnée de compétition attachée, GPU/internet off). |
| `notebook/INSTALL.md` | Cellule d'installation du notebook — **aucune dépendance externe**, stdlib seule. |
| `training/` | **Le pipeline d'entraînement réel** qui produit l'adaptateur 0.85 (SFT LoRA, backend Tinker). |
| `training/INSTALL.md` | Installation + variables d'environnement + ordre d'exécution du pipeline. |
| `requirements.txt` | Dépendances **directes**, versions exactes figées depuis `uv.lock`. |
| `requirements-lock.txt` | Environnement **complet** résolu (105 paquets), pour une repro au pin près. |

### `training/` en détail

```
train_sft.py            entraînement SFT LoRA rank-32 — le run 0.85
train_common.py         chargement du corpus, TrainingExample
loss_config.py          fonctions de perte (cross-entropy, IS, PPO, CISPO, DRO)
lr_schedule.py          schedule LR linéaire décroissante
reasoning.py            génération des CoT par famille (solveurs/reasoners)
augmentation.py         augmentation synthétique (cipher, bit_manipulation)
corpus.py               construction du corpus SFT (trace gardée ssi verify(gold, boxed))
convert_to_kaggle.py    Tinker -> format Kaggle (un-fuse experts MoE, merge Mamba gate+x_proj via QR+SVD)
package_for_kaggle.py   assemblage de submission.zip
upload_adapter.py       upload de l'adaptateur (Modal)
trainer/client.py       client du backend d'entraînement
symsolve/               reverse-engineering round-2 de la famille eq_symbol
  eqsym_re_round2.py    reproduit les expériences #17-#20 du writeup (offline)
pyproject.toml, uv.lock  spécification exacte de l'environnement
```

---

## Reproduire

**Le notebook** (rien à installer) :

```bash
jupyter notebook notebook/kaggle_writeup_notebook.ipynb
```
Il cherche `train.csv` sous `/kaggle/input/**/` ; hors Kaggle, place la donnée de
compétition à côté ou adapte la cellule 1.

**Le pipeline d'entraînement** : voir `training/INSTALL.md`.

**Les expériences #17-#20 du writeup** (eq_symbol) :

```bash
cd training
EQSYM_PROBLEMS_DIR=/chemin/vers/problems uv run python symsolve/eqsym_re_round2.py
```

---

## Notes sur ce lot

- Le `.ipynb` est fourni **non modifié**, à l'octet près, tel qu'exécuté et publié sur
  Kaggle — d'où un chemin de repli local en cellule 1, jamais atteint sur l'infra Kaggle.
- Deux chemins machine ont été rendus portables pour la diffusion (aucune clé n'a jamais
  été présente dans les fichiers) : le chemin en dur de la clé Tinker dans `train_sft.py`
  est devenu `TINKER_API_KEY` / `TINKER_API_KEY_FILE`, et le dossier `problems/` de
  `symsolve/eqsym_re_round2.py` est devenu `EQSYM_PROBLEMS_DIR` ; les dossiers de sortie
  de `convert_to_kaggle.py` / `package_for_kaggle.py` sont devenus `SUBMISSION_DIR`.
- Ne sont **pas** inclus, faute de taille : le corpus SFT (`corpus.jsonl`, ~42 Mo),
  les archives de soumission (~1,5-3,5 Go chacune) et les checkpoints d'adaptateurs.
  Ils restent sous `Competition Kaggle\` sur le disque source.

## Liens

- Notebook Kaggle : https://www.kaggle.com/code/lickel/reverse-engineering-the-ceiling
- Compétition : https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
