# 🎯 Playbook de Compétition — Reconnaissance → Stratégie → Combat

> **Mantra :** *On n'entre pas dans la bataille sans la carte. Le temps passé en reconnaissance n'est jamais perdu.*
>
> **Origine :** leçon tirée à nos dépens sur le NVIDIA Nemotron Challenge (puzzles `eq_symbol`). On a brute-forcé pendant des heures sous une **hypothèse de base fausse** (les symboles « base-26 », alors que c'étaient des **cryptarithmes décimaux**). La carte existait dès le départ — writeups gagnants, provenance, et une **famille sœur lisible** (`equation numeric`) qui révélait l'inventaire des règles. On a fait l'ordre inverse : combat → recherche → enfin la carte. Ce playbook remet l'ordre à l'endroit.

---

## Le fil rouge
**Recon → hypothèse validée → instance lisible comme clé → recherche falsifiable → effort aligné sur le prix.**

---

## Phase 0 — Reconnaissance (AVANT toute ligne de code)
*« Connais le terrain avant de te battre. »*
- [ ] **Lire le scoring / la métrique exacte** (le code de la métrique si fourni). C'est lui qui définit la victoire — pas l'élégance de la solution.
- [ ] **Provenance des données** : générateur public ? dataset card ? repo officiel ? édition précédente du concours ?
- [ ] **Prior art** : writeups gagnants, notebooks publics, discussions forum, solutions des éditions passées, papiers liés.
- [ ] **Cartographier** : familles/types de problèmes, tailles, format d'entrée/sortie, contraintes dures (budget compute, format de soumission, limites de temps).
- **Livrable :** 1 page de « carte » + une hypothèse de stratégie **écrite**.

## Phase 1 — Profilage des données pas cher (minutes, pas heures)
- [ ] Distributions (classes, longueurs de sortie, fréquences), invariants, cas dégénérés.
- [ ] Tests d'**élimination** rapides : quelles familles d'hypothèses tombent en quelques minutes ?
- **Règle :** aucune recherche lourde tant que les stats simples ne sont pas faites. Elles orientent (ou éliminent) des semaines de travail.

## Phase 2 — Interroger la représentation AVANT de construire
*« L'hypothèse de base non questionnée tue toute l'enquête. »*
- [ ] Lister les **hypothèses fondatrices** (encodage, modèle, unités, structure) et les **valider explicitement** sur quelques cas.
- [ ] Traiter toute hypothèse **héritée** (y compris la tienne, d'un travail précédent) comme **suspecte** jusqu'à preuve.
- **Drapeau rouge :** « tout le monde sait que c'est X » → vérifie X en premier.

## Phase 3 — L'instance la plus lisible d'abord, et comme clé
*La solution d'un groupe ouvre le suivant.*
- [ ] Identifier la version la plus **observable** du problème (ex. chiffres visibles vs chiffrés) ; la résoudre **en premier**.
- [ ] Réutiliser sa solution comme **clé** des variantes dures (la version en clair donne les règles de la version chiffrée).
- [ ] Agréger le **corpus entier** pour extraire les invariants partagés (vocabulaire borné, inventaire de règles…).

## Phase 4 — Recherche falsifiable + validation held-out
- [ ] **Held-out systématique** : valider sur des données que le fit n'a **pas** vues (pas seulement in-sample).
- [ ] **« Une solution cohérente ≠ LA solution »** : vérifier unicité + généralisation (méfiance des espaces sur-paramétrés : ils collent à tout).
- [ ] **Checkpoints de falsification** : si un cas *bien contraint* réfute l'hypothèse → **changer de modèle**, ne pas empiler des rustines (épicycles).

## Phase 5 — Aligner l'effort sur le chemin du prix
- [ ] Reverse-engineer **le chemin vers le score**, pas le problème intellectuellement le plus satisfaisant.
- [ ] Choisir tôt : **déterministe** / **modèle (ML/SFT/RL)** / **hybride** (déterministe sur le soluble, modèle sur le reste — souvent gagnant).

---

## 🧰 Discipline transversale
- **Instrumenter** : logs de progression, métriques intermédiaires.
- **Time-boxer** chaque piste ; fixer d'avance le critère d'abandon.
- **Échantillonner** avant tout run complet ; ne jamais piper un run long dans `tail` (sortie masquée jusqu'à la fin).
- **Journal des hypothèses** : `hypothèse → test → verdict`. Évite de tourner en rond et documente la frontière atteinte.

## ❌ Anti-patterns (les pièges où on est tombés)
- Foncer dans le brute-force avant la reconnaissance.
- Bâtir sur une hypothèse héritée jamais testée.
- Confondre « ça colle aux exemples » et « c'est la règle » (sur-ajustement / sous-détermination).
- Persister sur un modèle que les données réfutent déjà.
- Optimiser le problème intellectuel au lieu du score.

## ✅ Checklist Jour 1 (à cocher avant de coder)
1. [ ] Métrique comprise et relue.
2. [ ] Provenance + prior art ratissés (writeups, repos, éditions passées).
3. [ ] Données profilées (stats pas chères).
4. [ ] Hypothèses fondatrices listées **et validées**.
5. [ ] Instance la plus lisible identifiée (la « clé »).
6. [ ] Stratégie écrite : chemin vers le prix + premier jalon **falsifiable**.

---
*Ordre à respecter : carte d'abord, combat ensuite. C'est l'inverse qui nous a coûté.*
