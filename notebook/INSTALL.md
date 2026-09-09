# Cellule d'installation — notebook Kaggle

**Le notebook n'a aucune dépendance externe.** Il tourne sur l'image Kaggle par défaut
(CPU, internet désactivé) et n'importe que la bibliothèque standard :
`csv`, `re`, `sys`, `glob`, `math`, `random`, `collections`.

C'est délibéré : le notebook devait s'exécuter **vert** sur l'infra Kaggle avec
`"enable_internet": false` (cf. `kernel-metadata.json`), ce qui interdit tout `pip install`.

Si tu le rejoues hors Kaggle, la seule cellule d'installation nécessaire est :

```python
# aucune installation requise — Python >= 3.9 suffit
```

La seule ressource externe est la donnée de compétition, attachée via
`kernel-metadata.json` (`"competition_sources": ["nvidia-nemotron-model-reasoning-challenge"]`)
et trouvée par la cellule 1 sous `/kaggle/input/**/train.csv`.

> Le `.ipynb` est fourni **tel qu'exécuté et publié sur Kaggle**, non modifié — d'où le
> chemin de repli local `G:/...` en cellule 1, qui n'est jamais atteint sur Kaggle.

Pour les dépendances du **pipeline d'entraînement**, voir `../requirements.txt`.
