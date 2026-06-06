# H4-robot — Pre-registered Hypotheses

**Date de pré-registration**: 2026-06-06
**Auteur**: Axel Bellatalla (@reily157)
**Repo**: github.com/reily157/H4-robot
**Commit SHA**: [à compléter au moment du commit]

---

## Objectif et engagement

Ce document fixe **avant tout regard sur les données accumulées** les hypothèses que je m'engage à tester, les métriques précises, les seuils statistiques et les critères de rejet/acceptation.

Toute analyse exploratoire ultérieure non listée ici doit être marquée comme *exploratoire* et **ne peut pas être validée sur le dataset existant** — elle exigera des données futures pour confirmation, conformément aux standards de pré-registration scientifique.

Cette discipline vise à éviter trois pièges connus de la littérature (cf. synthèse `notes/lectures.md`) :

- **Look-ahead bias** — calibrer et tester sur les mêmes données
- **Data snooping rétroactif** — formuler des hypothèses après avoir vu les données et prétendre qu'elles étaient pré-établies
- **Multiple testing inflation** — augmenter mécaniquement le risque de faux positifs en multipliant les tests sans correction

---

## Contexte technique

**Dataset disponible au moment de la pré-registration :**

- 7 cycles d'opening BTC capturés en haute résolution sur Hyperliquid HIP-4
- Cycles complets : vendredi 29 mai, lundi 1, mardi 2, mercredi 3, jeudi 4, vendredi 5, samedi 6 juin 2026
- 6 fichiers JSONL par jour : `book_levels`, `trades`, `bbo`, `perp_ctx`, `raw_ctx`, `health`
- Latence sub-milliseconde (VPS Tokyo, même datacenter que les validators Hyperliquid)
- 44 subscriptions WebSocket actives (8 sides HIP-4 × 5 channels + BTC perp + outcomes catégoriels)

**Structure d'un cycle BTC HIP-4 :**

- 1 marché binaire "BTC above X at T" → 2 outcomes (YES/NO)
- 1 question multi-outcomes "BTC price range at T" → 3 outcomes catégoriels (Below threshold_low / In range / Above threshold_high) + 1 fallback
- Tous résolvent à 06:00 UTC quotidien
- Période d'opening auction : 15 minutes (06:00–06:15 UTC)
- Continuous trading ensuite

---

## Cadre statistique global

**Tous les tests d'hypothèses ci-dessous respectent :**

1. **Correction pour tests multiples** : Benjamini-Hochberg (FDR contrôlé à 5%) sur l'ensemble des hypothèses H1-H7 considérées simultanément. Bonferroni n'est pas utilisé car trop conservateur pour des hypothèses corrélées.

2. **Validation hors-échantillon** : pour les hypothèses qui le permettent (H1, H3, H4, H5, H7), validation par leave-one-cycle-out — chaque cycle est tour à tour exclu de la calibration et utilisé pour test out-of-sample.

3. **Bootstrap pour intervalles de confiance** : 10 000 réplications avec block bootstrap (Politis-Romano stationary bootstrap) pour respecter l'autocorrélation temporelle.

4. **Seuil de significativité brut** : α = 0.05 avant correction FDR.

5. **Coûts modélisés** : les hypothèses impliquant une exploitation potentielle utilisent un modèle de coûts réaliste — fees post-promo (0.025% maker / 0.05% taker estimés), slippage moyen 0.5% sur ordres > 50 USDH, coût d'opportunité du capital à 5% APR.

6. **Avertissement Manski/Wolfers-Zitzewitz** : toute déviation observée sur des prix < 0.20 ou > 0.80 est interprétée avec prudence car le mapping prix→probabilité y devient mécaniquement non-linéaire. Une déviation systémique aux extrêmes n'est pas automatiquement une inefficience.

---

## Hypothèses

### H1 — Invariant sum-to-1 sur les buckets catégoriels

**Énoncé** : La somme des mids des 3 outcomes nommés du bucket multi-outcomes (`bucket_idx_0` + `bucket_idx_1` + `bucket_idx_2`) doit converger vers 1.0 si le marché est efficient et risk-neutre.

**Métrique** :
- À chaque timestamp t où les 6 prix YES/NO des 3 outcomes sont simultanément disponibles, calcul de `dev(t) = (mid_0 + mid_1 + mid_2) - 1.0` où `mid_i = (bid_yes_i + ask_yes_i) / 2`.
- Distribution empirique de `dev(t)` sur les 15 premières minutes post-opening (06:00–06:15) puis sur la suite (06:15–24:00).

**Test statistique** : test de signe sur la médiane vs 0, test de Kolmogorov-Smirnov vs distribution centrée en 0.

**Critère de rejet de H1** :
- Médiane absolue |dev| > 0.5% au seuil α=0.05 corrigé FDR
- ET déviation persistante (autocorrélation ρ(1) > 0.3 sur lags de 10 secondes)

**Note interprétative** : un rejet de H1 ne signifie pas automatiquement un arbitrage — il peut refléter (a) risk-aversion non-neutre, (b) frais de résolution, (c) risque smart contract pricé en discount. Toute interprétation comme arbitrage devra valider que `1 - (sum of asks_yes_i)` > coûts totaux d'exécution.

---

### H2 — Volatilité implicite vs réalisée

**Énoncé** : La volatilité implicite (IV) extraite des prix du bucket multi-outcomes par inversion d'un modèle Black-Scholes adapté présente un écart systématique par rapport à la volatilité réalisée EWMA du BTC perp sur les 24h précédentes.

**Métrique** :
- À chaque timestamp t (sampling 1 minute), extraction de l'IV implicite à partir des trois prix `mid_0`, `mid_1`, `mid_2` du bucket, sachant les seuils `[threshold_low, threshold_high]` et le mark price BTC, en supposant une distribution log-normale du sous-jacent à l'expiry.
- Calcul de la RV (realized volatility) EWMA λ=0.94 du BTC perp `mark_px` sur la fenêtre [t-24h, t].
- Spread = IV(t) - RV(t).

**Test statistique** : régression OLS de `Spread(t)` sur constantes + time-to-expiry + activité de marché (volume des 5 dernières minutes). Test du coefficient constant ≠ 0.

**Critère de rejet de H2** : coefficient constant statistiquement ≠ 0 au seuil α=0.05 corrigé FDR, et magnitude > 1 point de volatilité annualisée.

**Note interprétative** : un spread IV-RV > 0 peut indiquer une vol risk premium positive (analogue aux options actions). Un spread < 0 serait plus inhabituel et mériterait investigation.

---

### H3 — Convergence non-monotone du bucket central

**Énoncé** : Le mid de l'outcome bucket "in range" (idx:1) ne converge pas monotoniquement vers sa résolution finale au fil du cycle. Il présente des cycles de divergence-convergence qui pourraient refléter une dynamique de croyances vs sous-jacent.

**Métrique** :
- Pour chaque cycle, série temporelle de `mid_1(t)` sur les 24h.
- Décomposition en tendance (rolling 1h) et résidus.
- Comptage des "reversals" (changement de signe de la dérivée lissée) par fenêtre d'1 heure.
- Comparaison avec la trajectoire du BTC perp `mark_px(t)` relative aux seuils du bucket.

**Test statistique** : test de Mann-Kendall pour monotonicité (null : monotone), bootstrap des résultats.

**Critère de rejet de H3** :
- Rejet du test de monotonicité au seuil α=0.05 corrigé FDR
- ET nombre médian de reversals par cycle > 4 (donc au moins une oscillation toutes les 6 heures)

**Note interprétative** : ce test est exploratoire. Une non-monotonicité confirmée pourrait suggérer une dynamique exploitable (achat-vente cyclique) mais demanderait validation hors-échantillon stricte.

---

### H4 — Asymétrie delta empirique vs modèle (FLB sur outcomes extrêmes)

**Énoncé** : Sur les outcomes extrêmes (`bucket_idx_0` "Below" et `bucket_idx_2` "Above"), le delta empirique (sensibilité du mid à un changement de BTC) diffère systématiquement du delta théorique calculé sous distribution log-normale. Plus précisément, **les longshots sont surpricés**, conformément au favorite-longshot bias documenté dans la littérature (Snowberg & Wolfers 2010).

**Métrique** :
- Sampling 1-minute des mids `mid_0`, `mid_2` et du BTC `mark_px`.
- Régression locale `Δmid_i = α_i + β_i × Δmark_px` sur fenêtres de 30 minutes.
- Comparaison `β_i_empirical` vs `β_i_theoretical` (delta Black-Scholes-bucket).
- Score de surpricing : `score(t) = mid_i(t) - mid_i_theoretical(t)` où `mid_i_theoretical` est calculé sous distribution log-normale avec vol implicite ATM.

**Test statistique** : test du signe sur `score(t)` pour outcomes avec mid < 0.10 (longshots Below) et mid > 0.90 réflexion (longshots Above transformés en `1 - mid`).

**Critère de rejet de H4** :
- Score médian statistiquement > 0 au seuil α=0.05 corrigé FDR pour les longshots (mid < 0.10)
- ET magnitude > 0.5% (en termes de surpricing absolu)

**Note interprétative** : si confirmé, valider que (a) le biais persiste hors-échantillon par leave-one-cycle-out, (b) le surpricing dépasse les coûts modélisés (fees + slippage), (c) le sens du biais est cohérent à travers les cycles. Le FLB hippique persiste sur 50 ans sans être profitable — même en cas de confirmation H4, le test profit ≠ test fait.

---

### H5 — Arbitrage cross-marché binary ↔ buckets

**Énoncé** : Il existe une relation algébrique entre le prix du marché binaire "BTC above target" et les prix des buckets. Spécifiquement, `prix_binary ≈ P(BTC > target)` qui peut être déduit des bucket thresholds. Des divergences > 2× les fees totales suggéreraient un arbitrage théorique.

**Métrique** :
- À chaque timestamp t où binary et buckets sont simultanément traders, calcul de :
  - `implied_binary_from_buckets(t)` selon la position du target binary par rapport aux thresholds du bucket
  - `actual_binary(t) = mid_binary(t)`
  - `divergence(t) = abs(implied - actual)`
- Comparaison à `2 × (taker_fee_binary + taker_fee_bucket)` (~0.2% en valeur conservative).

**Test statistique** : fraction du temps où `divergence(t) > 2 × fees` ; intervalle de confiance bootstrap.

**Critère de rejet de H5** :
- Fraction > 5% du temps avec divergence > seuil au seuil α=0.05 corrigé FDR
- ET durée médiane des fenêtres d'arbitrage > 30 secondes (suffisant pour exécuter à la main)

**Note interprétative** : avant de claim un arb, vérifier que les ordres limit BUY sur YES (du côté sous-pricé) et BUY sur NO (du côté sur-pricé) peuvent réellement être remplis aux prix observés. Le spread bid-ask peut éliminer l'arb apparent.

---

### H6 — Lag funding rate (priorité 2)

**Énoncé** : Les changements de `funding` du BTC perp précèdent les mouvements de prix des outcomes HIP-4 avec un lag détectable, suggérant une asymétrie d'information.

**Métrique** :
- Sampling 30-secondes de `funding(t)` (perp_ctx) et des mids HIP-4.
- Cross-correlation entre `Δfunding` et `Δmid` des outcomes binary à différents lags (−5 min à +5 min par pas de 30s).
- Identification du lag avec corrélation maximale.

**Test statistique** : test du lag optimal ≠ 0 par bootstrap.

**Critère de rejet de H6** :
- Lag optimal statistiquement ≠ 0 au seuil α=0.05 corrigé FDR
- ET |lag| > 30 secondes (différenciable du bruit)
- ET corrélation à ce lag > 0.10 en magnitude absolue

**Note interprétative** : Hypothèse marquée priorité 2 car le funding rate change peu sur 24h. Probabilité de signal faible mais test peu coûteux à exécuter.

---

### H7 — Opening auction overnight bias

**Énoncé** : Le mouvement de prix pendant l'auction d'ouverture (05:59 → 06:15 UTC) prédit partiellement le retour des 5 minutes suivantes (06:15 → 06:20 UTC). Si l'auction sous-réagit ou sur-réagit à l'information overnight, les premières minutes de continuous trading corrigent ce biais.

**Métrique** :
- Pour chaque cycle, calcul de :
  - `Δ_auction = mid_binary(06:15) - mid_binary(05:59)`
  - `Δ_postopen = mid_binary(06:20) - mid_binary(06:15)`
- Régression OLS `Δ_postopen = α + β × Δ_auction + ε` sur l'ensemble des cycles disponibles.

**Test statistique** : test du coefficient β ≠ 0 (test t Student, bootstrap pour robustesse).

**Critère de rejet de H7** :
- |β| > 0.15 statistiquement significatif au seuil α=0.05 corrigé FDR
- ET R² > 0.10 (au moins 10% de variance expliquée)
- ET le signe de β est cohérent à travers ≥ 5 des 7 cycles

**Note interprétative** : c'est l'hypothèse la plus prometteuse pour une stratégie exécutable (ordre à 06:15 dans le sens prédit par le mouvement d'auction). Néanmoins, avec n=7 cycles, la puissance statistique est faible — un signal positif demandera 3-6 mois de collecte supplémentaire pour confirmation avant tout déploiement de capital.

---

## Hypothèses exploratoires (NON pré-registrées)

Les hypothèses suivantes ne sont **pas** pré-registrées et **ne peuvent pas être validées sur le dataset actuel**. Elles sont notées ici pour mémoire et pourront faire l'objet d'une nouvelle pré-registration future si des collectes de données supplémentaires sont effectuées.

- **H8 (backlog)** — VPIN asymétrique par side. Décomposition des trades en aggressor BUY vs SELL sur YES et NO séparément ; calcul VPIN par side ; test d'asymétrie. Méthodologiquement complexe, à reporter.

- **Anomalies découvertes en regardant les données** — toute pattern observé en explorant les fichiers JSONL après cette pré-registration doit être noté comme exploratoire et ne peut **pas** être présenté comme validé sur ces données. La validation exigera des cycles futurs non encore capturés.

---

## Protocole d'analyse

**Ordre d'exécution** :

1. Charger les 7 fichiers JSONL en DuckDB pour requêtes analytiques (ingest read-only depuis les fichiers compressés `.jsonl.gz`)
2. Exécuter chaque test H1 à H7 indépendamment, produire les statistiques et IC bootstrap
3. Appliquer correction FDR Benjamini-Hochberg sur l'ensemble
4. Pour les hypothèses non rejetées : effectuer validation leave-one-cycle-out
5. Produire un rapport HTML/PDF avec figures et tableaux

**Critères de décision finale (après tous les tests)** :

- **Cas A — Aucune hypothèse non rejetée après FDR** : conclusion "pas d'anomalie détectée dans ce dataset"
- **Cas B — 1-2 hypothèses non rejetées avec validation OOS positive** : projet continue, on monitore la persistance sur des cycles futurs
- **Cas C — ≥ 3 hypothèses non rejetées et cohérentes entre elles** : signal probable, on envisage la phase d'exécution avec capital test 50 USDH selon protocole séparé (à définir)

**Engagement** : la décision finale est prise selon ces critères, indépendamment de l'attrait subjectif des résultats.

---

## Limites connues du protocole

- **Taille d'échantillon faible (n=7 cycles)** : puissance statistique limitée. Les hypothèses non rejetées doivent être considérées comme exploratoires en attente de réplication.
- **Période de marché unique** : tous les cycles sont en juin 2026, sur une seule semaine. Les conclusions ne sont pas généralisables à d'autres régimes de marché (volatilité différente, conditions différentes, etc.).
- **Latence d'exécution non testée** : tous les "arbitrages potentiels" identifiés sont théoriques. La capacité réelle d'exécuter à ces prix n'est pas validée sans paper trading.
- **Coûts post-promo non confirmés** : Hyperliquid offre actuellement 0% fees sur HIP-4. Les coûts modélisés (0.025% maker / 0.05% taker) sont des estimations conservatives basées sur les fees standards Hyperliquid perpetuals.
- **Pas de validation cross-marché** : le projet ne compare pas avec Polymarket, Kalshi, ou d'autres marchés de prédiction. Cette validation cross-marché serait un follow-up logique mais hors scope de ce protocole.

---

## Engagement de transparence

Tous les résultats — positifs comme négatifs — seront documentés et conservés dans le repo `H4-robot`. Les conclusions seront publiées dans un fichier `RESULTS.md` avec :

- Statistiques brutes de chaque test
- Intervalles de confiance bootstrap
- p-values brutes et p-values FDR-corrigées
- Figures et visualisations
- Discussion honnête des limites

L'objectif n'est pas de "trouver une opportunité" mais de **conduire une analyse scientifique rigoureuse** dont les conclusions, quelles qu'elles soient, sont documentées avec intégrité.