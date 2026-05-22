# Plan de Test — Modèles Students Performance (CatBoost)

**Auteurs** : Thafat KANEM & Yanis MORIN — EPSI Nantes 2026  
**Modèles** : CatBoostRegressor × 2 (Math Score, Language Mean Score)  
**Dataset** : StudentsPerformance_modified.csv (~1 000 élèves, 5 features catégorielles)

---

## Métriques de référence (jeu de test officiel, random_state=42)

| Modèle              | MAE     | RMSE    | R²     |
|---------------------|---------|---------|--------|
| Math Score          | 11.4542 | 14.3761 | 0.1507 |
| Language Mean Score | 10.3101 | 13.2784 | 0.2256 |

---

## UC1 — Prédiction du Math Score

**Description** : Estimer la note en mathématiques d'un élève à partir de ses caractéristiques socio-éducatives, dans le but d'identifier les élèves à risque ou de maximiser leur score.

| ID    | Scénario                                                                                  | Critère d'acceptation                          |
|-------|-------------------------------------------------------------------------------------------|------------------------------------------------|
| SC1.1 | Prédiction sur une entrée complète (toutes features renseignées)                          | Valeur prédite dans [0, 100]                   |
| SC1.2 | Prédiction sur le jeu de test officiel (random_state=42, 20% du dataset)                  | MAE < 12.0 et RMSE < 15.1                      |
| SC1.3 | Cohérence logique métier : élève avec  "lunch=standard " vs  "lunch=free/reduced "            | Score standard > score free/reduced            |
| SC1.4 | Cohérence logique métier : élève avec  "test preparation course=completed " vs  "none "       | Score completed > score none                   |
| SC1.5 | Cohérence agrégée : moyenne prédite sur tout le dataset par groupe lunch                  | Moy. standard > Moy. free/reduced              |

---

## UC2 — Prédiction du Language Mean Score

**Description** : Estimer la moyenne lecture/écriture d'un élève pour orienter les décisions pédagogiques en compétences linguistiques.

| ID    | Scénario                                                                                  | Critère d'acceptation                          |
|-------|-------------------------------------------------------------------------------------------|------------------------------------------------|
| SC2.1 | Prédiction sur une entrée complète                                                        | Valeur prédite dans [0, 100]                   |
| SC2.2 | Prédiction sur le jeu de test officiel                                                    | MAE < 10.9 et RMSE < 14.0                      |
| SC2.3 | Cohérence logique métier :  "test preparation course=completed " vs  "none "                  | Score completed > score none                   |
| SC2.4 | Cohérence agrégée : moyenne prédite sur tout le dataset par genre                        | Moy. female > Moy. male (corrélation connue)   |

---

## UC3 — Détection de profils inhabituels (Isolation Forest)

**Description** : Scorer chaque entrée pour estimer si elle appartient au domaine d'entraînement et communiquer le niveau de confiance de la prédiction.

| ID    | Scénario                                                                                  | Critère d'acceptation                          |
|-------|-------------------------------------------------------------------------------------------|------------------------------------------------|
| SC3.1 | Score d'anomalie d'un profil typique (mode de chaque feature)                             | Score ≥ seuil de robustesse (−0.6222)          |
| SC3.2 | Plage des scores retournés par  "score_samples "                                            | Tous les scores dans [−1, 0]                   |
| SC3.3 | Couverture de la zone de robustesse sur le jeu de test                                    | ≥ 85% des entrées dans la zone de robustesse   |
| SC3.4 | Slice testing : entrées à score d'anomalie très bas vs très élevé                        | Entrées basses (< −0.6) ont RMSE > entrées hautes |

---

## UC4 — Gestion des valeurs manquantes (Imputation)

**Description** : Le pipeline doit rester fonctionnel et produire des prédictions plausibles même lorsque certaines features sont manquantes, grâce à l'imputation  "most_frequent ".

| ID    | Scénario                                                                                  | Critère d'acceptation                          |
|-------|-------------------------------------------------------------------------------------------|------------------------------------------------|
| SC4.1 | Entrée avec 1 valeur manquante ( "lunch=NaN ")                                              | Prédiction math dans [0, 100]                  |
| SC4.2 | Entrée avec toutes les features manquantes                                                | Prédictions math et language dans [0, 100]     |
| SC4.3 | Dégradation RMSE après imputation progressive sur  "lunch " (feature la plus corrélée)     | Dégradation RMSE < 20%                         |
| SC4.4 | Dégradation RMSE après imputation progressive sur  "gender " (feature critique language)   | Dégradation RMSE < 20%                         |

---

## UC5 — Robustesse au bruit catégoriel

**Description** : Simuler des erreurs de saisie ou de collecte en remplaçant aléatoirement x% des valeurs d'une feature par une catégorie valide tirée au hasard. Le modèle doit rester stable à faible niveau de bruit.

| ID    | Scénario                                                                                  | Critère d'acceptation                          |
|-------|-------------------------------------------------------------------------------------------|------------------------------------------------|
| SC5.1 | 5% de bruit catégoriel sur  "lunch " → RMSE Math                                            | Variation RMSE < +15%                          |
| SC5.2 | 10% de bruit catégoriel sur  "lunch " → RMSE Math                                           | Variation RMSE < +15%                          |
| SC5.3 | 5% de bruit catégoriel sur  "gender " → RMSE Language                                       | Variation RMSE < +15%                          |
| SC5.4 | 10% de bruit catégoriel sur  "gender " → RMSE Language                                      | Variation RMSE < +15%                          |

---

## UC6 — Performance sous charge

**Description** : Le pipeline complet (imputation → encodage → Isolation Forest → prédiction) doit traiter des batches de grande taille dans les délais SLA définis.

| ID    | Scénario                                                              | Critère d'acceptation          |
|-------|-----------------------------------------------------------------------|--------------------------------|
| SC6.1 | Batch de 1 000 prédictions (les deux modèles)                         | Temps total < 5 secondes       |
| SC6.2 | Dataset complet (~1 000 entrées) — prédictions seules                 | Temps total < 2 secondes       |
| SC6.3 | Pipeline complet (IF inclus) sur dataset entier                       | Temps total < 5 secondes       |

---

## Structure des fichiers de test

 " " "
conftest.py             # Fixtures partagées (artifacts, dataset, split, profil typique)
utils.py                # Fonctions utilitaires (impute, run_if_pipeline)
test_unit.py            # UC1.1, UC2.1, UC3.2 — forme, type, plage
test_performance.py     # UC1.2, UC2.2           — MAE, RMSE
test_business_logic.py  # UC1.3–1.5, UC2.3–2.4, UC3.1 — logique métier
test_robustness.py      # UC3.3–3.4, UC4, UC5    — imputation, bruit, anomalies
test_load.py            # UC6                    — charge et latence
pytest.ini              # Markers : unit, perf, business, robustness, load
 " " "

---

## Commandes rapides

 " " "bash
pytest                        # Tous les tests
pytest -m unit                # Tests unitaires uniquement
pytest -m perf                # Tests de performance uniquement
pytest -m business            # Tests de logique métier
pytest -m robustness          # Tests de robustesse
pytest -m load                # Tests de charge
pytest -v --tb=short          # Verbose avec traces courtes
" " "
