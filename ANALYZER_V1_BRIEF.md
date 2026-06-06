# ANALYZER_V1_BRIEF

**Projet** : H4-robot — Offline analyzer for HIP-4 Hyperliquid prediction market data
**Version** : V1 (first iteration, tests H1-H7 from `hypotheses.md`)
**Author** : Axel Bellatalla
**Repo** : github.com/reily157/H4-robot
**Created** : 2026-06-06

---

## 1. Context

Le projet H4-robot a capturé 7 cycles d'opening BTC sur les marchés HIP-4 d'Hyperliquid entre le 29 mai et le 6 juin 2026. Les données sont stockées dans des fichiers JSONL append-only sur le VPS Tokyo, organisés par jour UTC.

Ce brief définit l'analyzer offline qui ingère ces données et teste les 7 hypothèses pré-registrées dans `hypotheses.md` (commit `f388d7e`).

**Ce brief est la spec technique. Le périmètre, les algorithmes et les critères statistiques viennent de `hypotheses.md`. Ce document décrit comment les implémenter en code.**

---

## 2. Scope and constraints

### In scope

- Charger les fichiers JSONL (compressés `.gz` et non compressés) dans DuckDB en mémoire pour requêtes analytiques rapides
- Implémenter les 7 tests H1-H7 selon les spécifications de `hypotheses.md`
- Appliquer les corrections statistiques (Benjamini-Hochberg FDR, bootstrap stationnaire)
- Validation hors-échantillon par leave-one-cycle-out
- Produire un rapport HTML autonome avec figures matplotlib/plotly
- Exporter les statistiques brutes en JSON pour reproductibilité

### Out of scope

- Aucun calcul en temps réel — analyzer purement offline
- Aucune logique d'exécution de trades — seulement analyse
- Aucune modification des fichiers JSONL — read-only strict
- Aucune connexion au WebSocket Hyperliquid
- Aucune logique de découverte de cycles — on lit ce qui est déjà capturé

### Discipline méthodologique stricte

- **Pas d'exploration libre du dataset au-delà des hypothèses H1-H7.** Toute découverte fortuite est marquée comme `exploratory` et ne compte pas comme validation.
- **Look-ahead bias** : pour toute opération de join entre deux séries temporelles, utiliser `direction='backward'` (jamais `'nearest'` ou `'forward'`).
- **Multiple testing** : appliquer Benjamini-Hochberg sur l'ensemble des p-values H1-H7 ensemble, pas séparément.
- **Bootstrap stationnaire** (Politis-Romano 1994) pour intervalles de confiance, pas iid simple.

---

## 3. Repository structure

```
analyzer/
├── __init__.py
├── ingest.py            # JSONL → DuckDB in-memory load
├── stats.py             # FDR, bootstrap, OOS validation utilities
├── hypotheses/
│   ├── __init__.py
│   ├── h1_sum_to_one.py
│   ├── h2_iv_vs_rv.py
│   ├── h3_bucket_convergence.py
│   ├── h4_delta_asymmetry.py
│   ├── h5_cross_market_arb.py
│   ├── h6_funding_lag.py
│   └── h7_auction_bias.py
├── plots.py             # Matplotlib/plotly figures
├── report.py            # HTML report generation
├── main.py              # Orchestration: ingest → tests → report
└── tests/
    ├── __init__.py
    ├── test_ingest.py
    ├── test_stats.py
    └── test_hypotheses/
        ├── test_h1.py
        ├── test_h2.py
        └── ... (one per hypothesis)
```

**Placement** : tout l'analyzer vit sous `/home/ubuntu/H4-robot/analyzer/`. Les modules existants (`store.py`, `recorder.py`, etc.) **ne sont pas modifiés**.

---

## 4. Dependencies

À ajouter dans `requirements.txt` si pas déjà présentes :

```
duckdb              # Already in repo (used by old store.py)
pandas              # DataFrame manipulation
numpy               # Numerical computation
scipy               # Statistical tests (Mann-Kendall, KS, etc.)
statsmodels         # OLS regression, FDR correction
matplotlib          # Static figures
plotly              # Interactive figures (optional, for HTML report)
jinja2              # HTML templating
arch                # Bootstrap stationnaire (Politis-Romano)
```

Aucune dépendance heavy nouvelle. Toutes sont stables et standard quant.

---

## 5. Detailed module specifications

### 5.1 `ingest.py`

**Rôle** : charger les fichiers JSONL (compressés ou non) dans une base DuckDB en mémoire ou sur disque temporaire.

**API publique** :

```python
def load_to_duckdb(data_dir: str, db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
    """
    Reads all JSONL files in data_dir, creates DuckDB tables.
    
    Tables created:
        - book_levels (ts_local, ts_remote, coin, side, level_idx, px, sz, n_orders)
        - trades (ts_local, ts_remote, coin, px, sz, side, tid)
        - bbo (ts_local, ts_remote, coin, bid_px, bid_sz, ask_px, ask_sz)
        - perp_ctx (ts_local, coin, mark_px, mid_px, oracle_px, funding, open_interest)
        - cycles (cycle_id, started_at, bucket_*, binary_*, raw_meta)
        - outcomes_map (cycle_id, outcome_id, role, yes_coin, no_coin, description)
    
    Indexes created on (coin, ts_local) for all event tables.
    Returns a DuckDB connection ready for queries.
    """
```

**Comportement** :
- Détecte automatiquement les fichiers `.jsonl` et `.jsonl.gz`
- Utilise `duckdb.read_json_auto()` pour les non-compressés
- Pour les `.gz` : décompresse en streaming via `gzip.open` puis ingest
- Convertit les timestamps ISO 8601 en `TIMESTAMPTZ`
- Crée des indexes sur `(coin, ts_local)` pour chaque table
- Logue le nombre de records chargés par table

**Tests** (`test_ingest.py`) :
- Charge un dossier de test avec quelques fichiers JSONL synthétiques
- Vérifie les counts par table
- Vérifie le mapping des colonnes (notamment les timestamps)

---

### 5.2 `stats.py`

**Rôle** : utilitaires statistiques transverses utilisés par toutes les hypothèses.

**API publique** :

```python
def fdr_correction(p_values: list[float], alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    """
    Benjamini-Hochberg FDR correction.
    Returns (rejected: list of bool, adjusted_p_values: list of float).
    """

def stationary_bootstrap(data: np.ndarray, statistic: callable, 
                         n_boot: int = 10_000, block_size: float = None) -> dict:
    """
    Politis-Romano stationary bootstrap for autocorrelated time series.
    Returns dict with: 'mean', 'std', 'ci_lower_95', 'ci_upper_95', 'all_samples'.
    """

def leave_one_cycle_out(cycles: list[str], hypothesis_test: callable) -> dict:
    """
    OOS validation: for each cycle, exclude it from calibration and test on it.
    Returns dict per cycle with results, plus aggregate consistency check.
    """

def merge_asof_safe(left: pd.DataFrame, right: pd.DataFrame, 
                    on: str, by: str = None) -> pd.DataFrame:
    """
    Wrapper around pd.merge_asof that ALWAYS uses direction='backward'
    to prevent look-ahead bias. Raises if 'on' column is not sorted.
    """
```

**Tests** (`test_stats.py`) :
- FDR : test avec p-values fictives connues, vérifier le résultat vs scipy/statsmodels reference
- Bootstrap : test sur série iid (résultat doit converger vers std analytique) et sur série AR(1) (résultat doit avoir CI plus large que iid)
- `merge_asof_safe` : doit lever une exception si on tente `direction='nearest'`

---

### 5.3 Hypothesis modules — pattern commun

Chaque hypothèse vit dans son fichier `analyzer/hypotheses/hX_*.py` et expose la même interface :

```python
def run(conn: duckdb.DuckDBPyConnection, cycles_to_test: list[str] = None) -> dict:
    """
    Execute the hypothesis test.
    
    Args:
        conn: DuckDB connection from ingest.load_to_duckdb()
        cycles_to_test: list of cycle_id to include. If None, use all cycles.
    
    Returns dict with:
        - 'hypothesis_id': str (e.g. 'H1')
        - 'description': str (short summary)
        - 'metric_values': dict (raw statistics)
        - 'p_value': float (raw, before FDR correction)
        - 'ci_lower_95': float
        - 'ci_upper_95': float
        - 'rejected_null': bool (at uncorrected alpha=0.05)
        - 'oos_consistency': dict (results of leave-one-cycle-out, if applicable)
        - 'interpretation_notes': list[str] (caveats from hypotheses.md)
        - 'figures': list[plt.Figure] (matplotlib figures for report)
    """
```

Ce pattern uniforme permet à `main.py` d'orchestrer sans connaître les détails internes.

---

### 5.4 Détails par hypothèse

Pour chacune, la spec implémentation. Tous les seuils et critères viennent de `hypotheses.md` — ce document ne fait que les opérationnaliser.

#### H1 — Sum-to-1 invariant

**Source de données** : table `bbo` filtrée sur les coins YES des 3 outcomes nommés du bucket pour chaque cycle (via la table `outcomes_map`).

**Algorithme** :
1. Pour chaque cycle, identifier les 3 paires (yes_coin, no_coin) des outcomes `bucket_idx_0`, `bucket_idx_1`, `bucket_idx_2`.
2. Récupérer les `bbo` snapshots de chacun des 6 coins (3 YES + 3 NO).
3. Joindre via `merge_asof_safe` sur le timestamp commun (tolérance 100ms).
4. Calculer `mid_i = (bid_yes_i + ask_yes_i) / 2` pour chaque outcome.
5. Calculer `dev(t) = (mid_0 + mid_1 + mid_2) - 1.0`.
6. Séparer les distributions opening (06:00-06:15) vs continuous (06:15-24:00).
7. Tests : signe sur médiane vs 0 (Wilcoxon signed-rank), KS vs N(0, σ̂).
8. Autocorrélation ρ(1) sur lags de 10 secondes pour vérifier persistance.

**Critère de rejet** (de hypotheses.md) :
- Médiane absolue |dev| > 0.5% au seuil α=0.05 corrigé FDR
- ET autocorrélation ρ(1) > 0.3

**Figures** :
- Histogramme de `dev(t)` (deux phases)
- Time series de `dev(t)` pour chaque cycle, avec lignes verticales aux thresholds 06:00 et 06:15
- Plot d'autocorrélation

#### H2 — IV vs RV

**Source de données** : `bbo` (bucket outcomes) + `perp_ctx` (BTC mark_px, mid_px).

**Algorithme** :
1. Pour chaque cycle, sampling 1-minute des mids du bucket et du BTC mark.
2. Implémentation d'un modèle Black-Scholes adapté : inverser numériquement σ telle que la distribution log-normale (S0=mark_px, T=time_to_expiry, σ=IV) assigne aux 3 régions [-∞, thresh_low], [thresh_low, thresh_high], [thresh_high, +∞] des probabilités égales aux 3 mids.
3. Sur la même fenêtre, calculer RV EWMA λ=0.94 du BTC perp sur les 24h précédant t.
4. Annualiser les deux mesures (× √(525600)).
5. Régression OLS `Spread = α + β1 × TTE + β2 × volume_5min + ε`.
6. Test du coefficient constant α via t-test.

**Notes implémentation** :
- L'inversion BS-bucket n'a pas de forme analytique : utiliser `scipy.optimize.brentq` sur une fonction objectif qui calcule l'erreur entre les probabilités modèle et les mids observés.
- Si l'inversion échoue (ex: mids incohérents → pas de σ qui matche), marquer ce timestamp comme NA et le compter dans les rapports.

**Critère de rejet** : constante OLS significativement ≠ 0 (p < 0.05 FDR) ET |α| > 1 point de vol annualisée.

**Figures** :
- Time series IV(t) vs RV(t) par cycle
- Scatter Spread vs TTE
- Distribution du Spread

#### H3 — Bucket convergence non-monotonicity

**Source de données** : `bbo` du coin YES de `bucket_idx_1` pour chaque cycle.

**Algorithme** :
1. Pour chaque cycle, série complète de `mid_1(t)`.
2. Lissage rolling 1h.
3. Dérivée discrète, comptage de changements de signe (reversals).
4. Test de Mann-Kendall pour monotonicité globale (null : monotone).
5. Bootstrap stationnaire pour CI sur le nombre médian de reversals.

**Critère de rejet** : test Mann-Kendall rejeté (p < 0.05 FDR) ET médiane reversals > 4 par cycle.

**Figures** :
- Trajectoire de `mid_1(t)` pour chaque cycle, avec marqueurs des reversals
- Trajectoire conjointe `mid_1(t)` et `mark_px(t)` normalisés

#### H4 — Delta asymmetry & FLB

**Source de données** : `bbo` des outcomes `bucket_idx_0` (Below) et `bucket_idx_2` (Above) + `perp_ctx`.

**Algorithme** :
1. Sampling 1-min de mid_0, mid_2 (YES coins) et mark_px.
2. Régression locale rolling 30min : `Δmid_i = α_i + β_i × Δmark_px + ε`.
3. Pour comparaison théorique, utiliser un modèle BS-bucket calibré sur ATM (mid_1 ≈ 0.5).
4. Calcul du score de surpricing : `score(t) = mid_i_observed(t) - mid_i_theoretical(t)`.
5. Filtrer les timestamps où `mid < 0.10` (longshots Below) ou `mid > 0.90` symétrisé.
6. Test du signe sur la médiane du score (Wilcoxon).

**Critère de rejet** : médiane score > 0 (p < 0.05 FDR) ET magnitude > 0.5%.

**Figures** :
- β_empirical vs β_theoretical par outcome et par cycle
- Distribution du score de surpricing
- Scatter mid_observed vs mid_theoretical

#### H5 — Cross-market arbitrage

**Source de données** : `bbo` du binary outcome + `bbo` des 3 outcomes nommés du bucket + `perp_ctx` (pour mark_px).

**Algorithme** :
1. Pour chaque cycle, identifier le binary target price et les bucket thresholds.
2. À chaque timestamp t :
   - Calculer `implied_binary_from_buckets(t)` selon la position du binary target par rapport aux thresholds (interpolation linéaire ou règle de découpe selon les cas)
   - `actual_binary(t) = (bid_yes_binary + ask_yes_binary) / 2`
   - `divergence(t) = |implied - actual|`
3. Définir seuil `2 × fees_total` = 0.2% conservateur.
4. Calculer fraction du temps avec divergence > seuil + durée médiane des fenêtres consécutives au-dessus du seuil.

**Critère de rejet** : fraction > 5% (p < 0.05 FDR sur bootstrap) ET durée médiane > 30s.

**Figures** :
- Time series implied_binary vs actual_binary par cycle
- Histogramme de divergence
- Distribution des durées de fenêtres au-dessus du seuil

#### H6 — Funding rate lag

**Source de données** : `perp_ctx` (BTC funding) + `bbo` du binary outcome.

**Algorithme** :
1. Sampling 30s de `funding(t)` et `mid_binary(t)`.
2. Différences `Δfunding(t) = funding(t) - funding(t-1)`, `Δmid(t)`.
3. Cross-correlation à différents lags (-5min à +5min, pas 30s).
4. Identification du lag avec |corrélation| maximale.
5. Bootstrap stationnaire sur les corrélations à chaque lag.

**Critère de rejet** : lag optimal ≠ 0 (p < 0.05 FDR), |lag| > 30s, |corrélation| > 0.10.

**Figures** :
- Cross-correlation plot (lag sur x, corr sur y)
- Time series Δfunding et Δmid par cycle

#### H7 — Opening auction overnight bias

**Source de données** : `bbo` du binary outcome aux timestamps précis 05:59, 06:15, 06:20 pour chaque cycle.

**Algorithme** :
1. Pour chaque cycle :
   - Extraire les mids aux 3 timestamps (avec tolérance ±5s, prendre le snapshot le plus proche)
   - Calculer `Δ_auction = mid(06:15) - mid(05:59)`
   - Calculer `Δ_postopen = mid(06:20) - mid(06:15)`
2. Régression OLS sur les n=7 paires (Δ_auction, Δ_postopen) : `Δ_postopen = α + β × Δ_auction + ε`.
3. Test du coefficient β ≠ 0 (t-test Student) + bootstrap par leave-one-cycle-out.
4. Vérifier consistance du signe de β à travers les cycles (≥ 5/7 même signe pour validation).

**Critère de rejet** :
- |β| > 0.15 (p < 0.05 FDR)
- R² > 0.10
- Signe cohérent sur ≥ 5 cycles

**Figures** :
- Scatter Δ_auction vs Δ_postopen avec droite de régression
- Bar chart des β estimés par leave-one-cycle-out
- Time series du binary mid pendant l'opening pour chaque cycle (alignées sur 06:00 UTC)

**Note importante** : H7 est marquée dans `hypotheses.md` comme la plus prometteuse pour stratégie exécutable, MAIS avec n=7 cycles la puissance statistique est faible. Tout signal positif devra être confirmé par 3-6 mois de collecte supplémentaire avant déploiement de capital.

---

### 5.5 `report.py` & `plots.py`

**Rôle** : produire un rapport HTML autonome qui agrège les résultats de toutes les hypothèses.

**Format de sortie** :

```
output/
├── report.html               # Rapport principal
├── report.json               # Statistiques brutes pour reproductibilité
└── figures/
    ├── h1_*.png
    ├── h2_*.png
    └── ...
```

**Structure du HTML** :

1. Header : timestamp d'analyse, hash du commit `hypotheses.md`, dataset summary (nb cycles, période, nb total events par table)
2. Pour chaque hypothèse :
   - Énoncé (depuis hypotheses.md)
   - Statistiques brutes et FDR-corrigées
   - Verdict : `REJECTED` / `NOT REJECTED` / `INCONCLUSIVE`
   - Figures intégrées
   - Notes interprétatives
3. Synthèse finale : table récapitulative + classification dans cas A/B/C selon hypotheses.md.

**Templating** : Jinja2 pour le HTML, matplotlib pour les figures statiques (PNG dans le HTML), optionnellement plotly pour interactif (à voir si nécessaire).

---

### 5.6 `main.py`

**Orchestration** :

```python
def main(data_dir: str, output_dir: str, cycles_filter: list[str] = None):
    """
    1. Load data via ingest.load_to_duckdb()
    2. Run all 7 hypothesis tests
    3. Apply FDR correction across all p-values
    4. Run OOS validation where applicable
    5. Generate HTML report + JSON dump + figures
    """
```

**Exécution depuis terminal** :
```bash
cd /home/ubuntu/H4-robot
python -m analyzer.main --data-dir data/ --output-dir output/
```

---

## 6. Validation criteria

**Avant que l'analyzer soit considéré "ready"** :

1. **Tous les tests unitaires passent** : `pytest analyzer/tests/ -v`
2. **L'ingest fonctionne** sur le dataset réel : charge tous les `.jsonl` et `.jsonl.gz` sans erreur, avec counts cohérents
3. **Chaque module H1-H7 retourne un dict respectant l'interface**, même si les résultats sont NA (par exemple si pas assez de données pour le test)
4. **Le rapport HTML est généré** sans erreur et est lisible visuellement
5. **Reproductibilité** : `report.json` permet de reconstruire tous les chiffres sans relancer l'analyse

---

## 7. Phasing

**Phase 1 — Foundations** (priorité absolue)
- `ingest.py` + tests
- `stats.py` + tests
- Charger les vraies données et vérifier que les counts sont cohérents

**Phase 2 — Hypothesis modules**
- Implémenter H1 et H7 en premier (les plus simples et les plus importantes)
- Tests unitaires pour chaque
- Validation que les outputs respectent l'interface commune

**Phase 3 — Hypothesis modules (suite)**
- H3, H5 (plus simples que H2/H4)
- H2, H4 (nécessitent inversion BS-bucket, plus complexes)
- H6 en dernier (priorité 2 dans hypotheses.md)

**Phase 4 — Reporting**
- `report.py` + `plots.py`
- Format JSON brut
- Templating Jinja2

**Phase 5 — End-to-end run**
- `main.py` orchestration
- Run complet sur le dataset
- Inspection des résultats
- Documentation finale dans `RESULTS.md`

**Validation entre phases** : valide explicitement avec moi avant de passer à la phase suivante. Pas de cascade de fixes sans accord.

---

## 8. Anti-patterns à éviter

Liste explicite des erreurs déjà rencontrées dans le projet, à ne pas reproduire :

1. **Pas de cascade de "petits fixes" sans validation** — chaque modification est commitée séparément avec son test
2. **Pas d'application de modifications "tant qu'on y est"** — scope strict du commit en cours
3. **Pas de `merge_asof(direction='nearest')`** — toujours `'backward'` (utilise `merge_asof_safe()`)
4. **Pas de p-values brutes présentées comme significatives** — toujours FDR-corrigé
5. **Pas de "j'ai vu ce pattern dans les données donc je le teste"** — toute exploration post-hoc est marquée exploratoire et exclue de la validation principale
6. **Pas de "le test est marginal mais ça pourrait être intéressant"** — les seuils de rejet sont fixés dans `hypotheses.md`, on s'y tient
7. **Pas de modification de `hypotheses.md` après ce commit** — si une hypothèse mérite ajustement, elle devient une nouvelle hypothèse exploratoire à valider sur données futures

---

## 9. Open questions

Questions volontairement laissées ouvertes — à discuter avec moi avant implémentation :

- **Format des cycles_to_test** : liste de cycle_id complets (ex: `["BTC_202605290600", ...]`) ou par date courte ?
- **Plotly vs matplotlib pour figures interactives** : matplotlib only au début (plus simple), plotly en V2 si utile
- **Performance** : pour 7 cycles × millions d'events, le bottleneck sera probablement l'ingest. À mesurer ; si > 5 minutes, optimiser via DuckDB COPY ou parquet intermédiaire.
- **Granularité du sampling** : H2 et H6 utilisent du 1-min / 30s. Confirmer que c'est suffisant ou si on a besoin de descendre plus fin.

---

## 10. Reminder of project values

Citation directe depuis `hypotheses.md` :

> "L'objectif n'est pas de 'trouver une opportunité' mais de conduire une analyse scientifique rigoureuse dont les conclusions, quelles qu'elles soient, sont documentées avec intégrité."

L'analyzer respecte cet engagement. Si tous les tests rejettent les hypothèses, c'est un résultat — et le rapport doit le présenter avec autant de soin qu'un résultat positif.