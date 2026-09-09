# NVIDIA Nemotron Model Reasoning Challenge — pipeline, writeup et journal

*English: this repository contains the full pipeline and the published writeup behind a bronze-medal solo entry (261st / 4,182 teams) to the NVIDIA Nemotron Model Reasoning Challenge on Kaggle. It was produced by AI models under my direction, as an exercise in AI piloting — see "How this was built" below.*

**Résultat :** médaille de bronze, 261e sur 4 182 équipes, en solo. Score 0,85 sur le leaderboard, au niveau de la baseline publique, sur un pipeline entièrement reconstruit et contrôlé.

- Writeup publié : [`WRITEUP.md`](WRITEUP.md) — *Reverse-engineering the ceiling: a controlled 0.85 and a reproducible frontier map*
- Notebook Kaggle : https://www.kaggle.com/code/lickel/reverse-engineering-the-ceiling
- Compétition : https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge
- Inventaire des fichiers et instructions de reproduction : [`LIVRABLES.md`](LIVRABLES.md)

## Comment ce dépôt a été construit

Je ne suis pas ingénieur en apprentissage automatique. Je viens de vingt ans d'industrie et je me suis reconverti vers l'IA en apprenant à la diriger plutôt qu'à la coder.

Ce dépôt est le produit de cette méthode. Le code, les expériences et le writeup ont été **produits par des modèles d'IA sous ma direction** : j'ai cadré le problème, décomposé le travail, arbitré entre les pistes, exigé des tests et un journal d'expériences, vérifié les résultats contre la métrique officielle, et décidé de ce qui partait en soumission. Vingt expériences sont consignées dans le writeup, échecs compris.

Ce que je comprends du pipeline : son architecture, la logique des solveurs déterministes qui fabriquent les données d'entraînement, pourquoi une trace de raisonnement n'est gardée que si sa réponse est vérifiée, pourquoi l'évaluation doit se faire à la limite exacte du harnais, et ce que veut dire « résoluble hors ligne n'est pas apprenable ».

Ce que je ne prétends pas maîtriser : les détails mathématiques de la conversion d'adaptateur (fusion des projections Mamba par QR + SVD), les fonctions de perte au-delà de l'entropie croisée, et l'écriture de ces composants sans assistance. Je le dis ici pour que le dépôt soit lu pour ce qu'il est.

Le [`COMPETITION_PLAYBOOK.md`](COMPETITION_PLAYBOOK.md) est le carnet de bord tiré de cette compétition : la méthode que j'applique désormais avant toute ligne de code.

## Ce que le dépôt ne contient pas

Le corpus SFT (~42 Mo), les archives de soumission (1,5 à 3,5 Go chacune) et les checkpoints d'adaptateurs ne sont pas inclus, faute de taille. Aucune clé d'API n'a jamais figuré dans ces fichiers ; les chemins machine ont été remplacés par des variables d'environnement (voir `training/INSTALL.md`).

## Licence

MIT — voir `LICENSE`.
