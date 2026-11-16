# fl-ids-web : détection d'intrusions applicatives par apprentissage fédéré

Prototype du TP « Architecture de détection des intrusions applicatives basée sur le Federated Learning
pour les applications web ».

**Auteurs :** AGBESSI John et ADJOVI Rubens
**École :** IFRI (Institut de Formation et de Recherche en Informatique), Université d'Abomey-Calavi
**Enseignant :** Dr ZANNOU Boris

Idée centrale : plusieurs organisations apprennent ensemble à reconnaître les attaques web (injection SQL, XSS,
traversée de répertoires, injection de commandes) **sans qu'aucun journal de requêtes ne quitte son périmètre**.
Seules des mises à jour de modèle, éventuellement bruitées et masquées, sont échangées.

Le code est en Python et NumPy pur (rétropropagation écrite à la main) : on voit exactement ce qui circule entre
les participants et le serveur, ce qui permet de tester l'écrêtage, le bruit, le masquage et l'agrégation.

## Démarrage rapide

```bash
pip install -r requirements.txt
python -m pytest -q                      # 21 tests
python -m fl_ids demo --dp --secagg --secure-filter   # une exécution complète, registre de modèles inclus
python -m fl_ids detect "GET /search?q=' OR 1=1--"    # scorer une requête avec le modèle exporté
python -m fl_ids all --seeds 3                        # toutes les expériences, figures et results/RESULTS.md
```

## Correspondance avec l'architecture du TP

| Section du TP | Module | Ce qui est implémenté |
|---|---|---|
| 5.2 Prétraitement local | `fl_ids/features.py` | Masquage (jetons, mots de passe, IBAN, e-mails) en conservant la forme, 18 caractéristiques syntaxiques, trigrammes hachés vers un **vocabulaire global fixe** (128 cases) |
| 5.3 Étiquettes | `fl_ids/synthetic.py` | Peu d'attaques étiquetées par participant, trafic bénin majoritaire, familles absentes localement |
| 5.4 Modèle | `fl_ids/model.py` | MLP supervisé + autoencodeur, scores fusionnés (« ou bruité ») |
| 6.1 Entraînement local | `fl_ids/node.py` | Époques locales limitées, terme proximal FedProx |
| 6.2 Agrégation | `fl_ids/server.py`, `aggregation.py` | FedAvg pondéré, FedProx, médiane, moyenne tronquée |
| 6.3 Non-IID | `fl_ids/node.py` | Tête de classification affinée localement, seuil de l'autoencodeur calibré localement |
| 7.2 Confidentialité différentielle | `fl_ids/model.py`, `privacy.py` | DP-SGD (écrêtage par exemple + bruit gaussien), comptabilité de Rényi avec sous-échantillonnage |
| 7.2 Agrégation sécurisée | `fl_ids/secagg.py` | Masquage additif par paires, les masques s'annulent dans la somme |
| 7.2.1 Cycle de vie du budget | `fl_ids/privacy.py` | Fenêtre glissante : au-delà du plafond, base gelée, seule la tête locale évolue |
| 7.3 Empoisonnement | `fl_ids/aggregation.py` | Filtre d'intégrité sur statistiques seules (norme, cosinus à la médiane), compatible SecAgg |
| 8 Gouvernance | `fl_ids/server.py` | Registre versionné (SHA-256), validation distribuée (rappel central + vote des nœuds sur les faux positifs), rollback |
| 9 Détection en ligne | `fl_ids/cli.py` | Seuils gradués : journaliser, alerter, bloquer |
| 10 Évaluation | `fl_ids/experiments.py` | Local seul / fédéré / centralisé, empoisonnement, compromis DP |

## Résultats (3 graines, 5 participants, 20 rounds)

Détail complet, avec figures, dans [`results/RESULTS.md`](results/RESULTS.md).

| Configuration | F1 | Rappel (connues) | Rappel (inédites) | Faux positifs |
|---|---|---|---|---|
| Local seul | 0,930 | 0,907 | 0,902 | 1,4 % |
| Fédéré (FedAvg) | 0,960 | 0,976 | 0,891 | 1,2 % |
| Fédéré + DP-SGD + SecAgg | 0,944 | 0,948 | 0,921 | 1,7 % |
| Centralisé (borne haute) | 0,961 | 0,981 | 0,885 | 1,2 % |

- Le modèle fédéré se situe nettement au-dessus du modèle local et au niveau du centralisé, comme le prévoit le TP.
- Le gain vient des familles d'attaques absentes des étiquettes locales. Sur les attaques **inédites**
  (SSTI, Log4Shell, XXE, NoSQLi), le gain de la fédération est nul : c'est l'autoencodeur qui les détecte, et il
  n'apprend que le trafic normal.
- Avec 1 participant malveillant sur 5 (mise à jour inversée et amplifiée ×10), FedAvg s'effondre (F1 ≈ 0,58). La
  validation avec rollback **seule ne suffit pas** (F1 ≈ 0,61). La médiane, la moyenne tronquée et le filtre d'intégrité compatible SecAgg ramènent le F1 à 0,95 environ.
- DP-SGD : σ = 1,0 donne ε ≈ 15 pour une perte de F1 d'environ 2 points, σ = 2,0 donne ε ≈ 5 pour une perte de
  9 points. Avec seulement 5 silos et peu de données par silo, une garantie forte coûte cher en précision.

## Limites assumées

- **Données synthétiques.** Les participants et les attaques sont simulés (`fl_ids/synthetic.py`). Les jeux
  recommandés par le TP (HTTPCS-IDS, CIC-IDS2018, WAF-AMiner, InSDN) ne sont pas intégrés : les scores absolus ne
  se transposent pas au trafic réel. Seule la comparaison entre configurations est informative.
- **SecAgg simulé.** Flottants au lieu d'un corps fini, pas de gestion des abandons, accord de clés simulé par une
  graine de session.
- **Filtre d'intégrité = fonctionnalité idéale.** Le serveur ne reçoit que la norme et le cosinus de chaque mise à
  jour, mais ces statistiques sont calculées en clair dans le processus. Le chiffrement fonctionnel ou les preuves à
  divulgation nulle de connaissance qui les rendraient réellement vérifiables ne sont pas implémentés.
- **Fenêtre glissante du budget ε.** Elle n'a de valeur formelle que si chaque fenêtre utilise une tranche de
  données distincte. Sinon les budgets se composent.
- **DP-SGD.** Lots de taille fixe tirés par permutation, alors que la comptabilité suppose un échantillonnage de
  Poisson (approximation courante). Le MLP et l'autoencodeur lisent les mêmes enregistrements : leurs coûts sont
  additionnés. La comptabilité est validée sur l'exemple MNIST de TensorFlow Privacy (ε ≈ 3,0).
- **Normalisation.** Le TP suggère une normalisation locale des caractéristiques. Le prototype utilise à la place un
  standardiseur global fixé à la genèse du consortium, pour que le serveur puisse évaluer le rappel central.
- **Pas de session.** Les caractéristiques de comportement par session (cadence, taux d'erreurs 4xx/5xx) ne sont
  pas modélisées : la détection est faite requête par requête.

## Structure

```
fl_ids/
  synthetic.py    trafic HTTP synthétique, participants non-IID
  features.py     masquage + caractéristiques + vocabulaire fixe
  model.py        MLP, autoencodeur, DP-SGD
  node.py         participant : entraînement local, personnalisation
  server.py       rounds, validation, rollback, registre
  aggregation.py  FedAvg, médiane, moyenne tronquée, filtre d'intégrité
  secagg.py       agrégation sécurisée (simulation)
  privacy.py      comptabilité de Rényi, fenêtre de budget
  experiments.py  expériences ; report.py : figures et tableaux
  cli.py          python -m fl_ids ...
tests/            21 tests (gradient numérique, comptabilité, SecAgg, empoisonnement, gel du budget)
results/          JSON, figures, RESULTS.md, modèle exporté
```
