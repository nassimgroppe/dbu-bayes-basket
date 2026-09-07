[README.md](https://github.com/user-attachments/files/31907712/README.md)
# DBU Bayes Basket

**Machine Learning zur Konstruktion eines Aktienkorbs im DAX/MDAX — Eine methodische Studie zur Effizienzmarkthypothese**

Studienarbeit im Modul *Data Analyst I* an der Digital Business University of Applied Sciences (DBU) Berlin.

---

## Autor

**Nassim Groppe**
B.Sc. Data Science & Management, DBU Berlin
Abgabe: Oktober 2026

---

## Kernergebnis in zwei Sätzen

Ein einziger methodischer Fehler kehrte in der Voruntersuchung das Vorzeichen des Ergebnisses um — von +485 % scheinbarer Outperformance zu −5,76 Prozentpunkten realer Underperformance. Zwei strukturell verschiedene ML-Modellklassen (Random Forest und Neural Network) bestätigen unabhängig voneinander die schwache Effizienzmarkthypothese für den deutschen Aktienmarkt.

---

## Forschungsfrage

Kann eine datengetriebene Basket-Konstruktion aus DAX- und MDAX-Aktien durch Machine-Learning-Methoden eine belegbare Outperformance gegenüber einem passiven Equal-Weight-Benchmark erzielen — und ist dieses Ergebnis robust gegenüber der Modellwahl?

Die Frage wird in drei Hypothesen zerlegt:

- **H₁ (Prognose):** Prognostiziert das Modell die Wochenrendite besser als eine naive Mittelwert-Baseline?
- **H₂ (Strategie):** Erzielt der konstruierte Basket eine statistisch signifikante Outperformance gegenüber dem Benchmark?
- **H₃ (Methode):** Ist Bayesian Optimization der klassischen Grid Search überlegen?

---

## Datenbasis

| Dimension | Wert |
|-----------|------|
| Aktien | 80 (DAX-40 + MDAX-50 nach Lückenfilter) |
| Zeitraum | 2019-01-02 bis 2025-12-30 |
| Beobachtungen | 120.480 Ticker-Tag-Kombinationen |
| Handelstage | 1.779 |
| Features | 8 technische Indikatoren |
| Target | Forward_Return_1W (5 Handelstage, `shift(-5)`) |
| Datenquelle | yfinance-API (Yahoo Finance) |

**Features:** Daily_Return, Momentum_3M, Momentum_6M, Momentum_12M, Volatility_30d, MA_50, MA_200, RSI_14

**Split:** Training bis 2023-12-31 · Test ab 2024-01-01 · **Strikte Hold-Out-Trennung**

---

## Methodik im Überblick

### Modell-Pipeline

1. **Datenbereinigung** (Notebook 01, 11) — Missing-Value-Behandlung, Ticker-Filter
2. **Feature-Engineering** (Notebook 11) — Momentum, Volatilität, technische Indikatoren
3. **Clustering** (Notebook 14) — K-Means mit K=6 zur Diversifikation
4. **Modellierung** — zwei parallele Modellklassen:
   - **Random Forest** mit Bayesian Optimization (Notebook 15)
   - **Neural Network** mit Bayesian Optimization (Notebooks 17, 18)
5. **Hold-Out-Korrektur** (Notebook 15b) — strikte zeitliche Trennung
6. **Backtest** — wöchentliches Rebalancing (Notebooks 16, 19)

### Warum zwei Modellklassen?

Die Erweiterung um ein neuronales Netz dient der **Robustheitsprüfung**. Ein Ergebnis, das über zwei strukturell verschiedene Modellfamilien persistiert — ein baumbasiertes Ensemble und ein tiefes neuronales Netz — ist wissenschaftlich belastbarer als ein Einzelmodell-Ergebnis.

---

## Zentrales Ergebnis — Der methodische Wendepunkt

| Zustand | Metrik | Wert |
|---------|--------|------|
| **t₀** — Mit Look-Ahead-Bias | Total Return | **+485 %** |
| **t₁** — Nach Korrektur (RF) | Differenz zum Benchmark | **−5,76 pp** |
| **t₁** — Nach Korrektur (NN) | Differenz zum Benchmark | **−27,45 pp** |

Der Fehler verzerrte nicht die Größenordnung — er kehrte das Vorzeichen um.

---

## Backtest-Ergebnisse — Modellvergleich

Zeitraum: 08.01.2024 bis 19.11.2025 (477 Handelstage · 96 Rebalancings)

| Kennzahl | RF-Basket | NN-Basket | Equal-Weight-Benchmark |
|----------|-----------|-----------|------------------------|
| Total Return | +9,18 % | +3,28 % | +30,73 % |
| Annualisierte Rendite | +4,72 % | +1,43 % | +12,53 % |
| Annualisierte Volatilität | 25,09 % | 23,67 % | 15,68 % |
| Sharpe Ratio | 0,31 | 0,18 | 0,83 |
| Sortino Ratio | 0,51 | 0,26 | 1,21 |
| Max Drawdown | −28,11 % | −26,78 % | −21,34 % |
| p-Wert (t-Test) | 0,9496 | 0,6407 | — |
| Signifikant (α = 0,05) | Nein | Nein | — |

**Beide Modellklassen unterliegen dem passiven Benchmark in jeder gemessenen Dimension. Die Renditedifferenzen sind wirtschaftlich substanziell, aber statistisch nicht signifikant.**

---

## Bayesian Optimization — Ergebnis-Muster

Ein besonders aufschlussreiches Ergebnis ergab die systematische Hyperparameter-Optimierung des neuronalen Netzes:

| Metrik | Baseline NN | Nach 30 Bayes-Iterationen |
|--------|-------------|---------------------------|
| Out-of-Sample R² | −0,0107 | **−0,0168** |

Die Optimierung verschlechterte die Out-of-Sample-Performance marginal, obwohl der Cross-Validation-Score sich verbesserte. Der Algorithmus wählte die kapazitätsstärkste Architektur (256-128-64 Neuronen) bei minimaler Regularisierung — ein klassisches Signal für Overfitting nach López de Prado (2018).

**Interpretation:** Bei niedrigem Signal-zu-Rausch-Verhältnis in Aktienrenditen begünstigt intensivere Modelloptimierung Overfitting auf Cross-Validation-Splits ohne reale Prognoseverbesserung. Diese Beobachtung ist der eigentliche methodische Ertrag der NN-Erweiterung.

---

## Theoretische Einordnung

Die Ergebnisse stützen die **schwache Form der Effizienzmarkthypothese** nach Fama (1970):

> Alle relevanten Informationen aus historischen Preisen sind bereits in den aktuellen Marktpreisen eingepreist. Aus einer Vergangenheit von Preisdaten lassen sich keine systematischen Überrenditen ableiten.

Da die verwendeten Features (Momentum, Volatilität, technische Indikatoren) ausschließlich preisbasiert sind, widerlegt das negative Ergebnis Fama nicht — es bestätigt ihn. Für eine potenzielle Widerlegung wären Informationen jenseits historischer Kurse nötig: Fundamentaldaten mit Point-in-Time-Historie, Sentiment-Signale, Cross-Asset-Faktoren.

---

## Repository-Struktur

```
dbu-bayes-basket/
├── notebooks/
│   ├── 00 DATA ACQUISITION DAX/MDAX.ipynb
│   ├── 01 DATA CLEANING.ipynb
│   ├── 02 EDA.ipynb
│   ├── 07 BASKET PCA CLUSTERING.ipynb
│   ├── 11 BASKET DATA CLEANING.ipynb
│   ├── 12 BASKET EDA.ipynb
│   ├── 13 BASKET MODELLING.ipynb
│   ├── 14 BASKET CLUSTERING.ipynb
│   ├── 15 BAYES BASKET OPTIMIZATION.ipynb
│   ├── 15b OUT-OF-SAMPLE-PREDICTIONS BAYES BASKET.ipynb
│   ├── 16 BASKET BACKTESTING.ipynb
│   ├── 17 NEURAL NETWORK BASELINE.ipynb           ← NN-Erweiterung
│   ├── 18 BAYES OPTIMIZATION NEURAL NETWORK.ipynb ← NN-Erweiterung
│   └── 19 NEURAL NETWORK BACKTEST.ipynb           ← NN-Erweiterung
├── data/
│   ├── raw/
│   │   └── dax_mdax_prices_2019_2025.parquet
│   └── processed/
│       ├── basket_features.parquet
│       ├── clusters.parquet
│       ├── predictions_bayes_oos.parquet
│       ├── predictions_nn_bayes_oos.parquet
│       ├── prices_cleaned.parquet
│       └── comparison_rf_nn_bench.csv
├── models/
│   ├── nn_baseline.pt
│   ├── nn_bayes_optimized.pkl
│   └── scaler_nn.pkl
├── images/
│   └── nn_backtest_comparison.png
├── requirements.txt
└── README.md
```

---

## Reproduzierbarkeit

### Voraussetzungen

- Python 3.9+
- Virtuelle Umgebung (venv oder conda)
- Ca. 2 GB freier Speicherplatz

### Installation

```bash
git clone https://github.com/nassimgroppe/dbu-bayes-basket.git
cd dbu-bayes-basket
python3 -m venv .venv
source .venv/bin/activate            # macOS/Linux
# .venv\Scripts\activate              # Windows
pip install -r requirements.txt
```

### Ausführungsreihenfolge

Die Notebooks bauen aufeinander auf und sollten in nummerischer Reihenfolge ausgeführt werden. Zwischenergebnisse werden im `data/processed/`-Verzeichnis persistiert.

**Kernpipeline Random Forest:**
`00 → 01 → 11 → 14 → 15 → 15b → 16`

**Kernpipeline Neural Network (Erweiterung):**
`17 → 18 → 19` *(setzt vorhandene Feature- und Cluster-Datensätze voraus)*

### Reproduzierbarkeit der Ergebnisse

Alle Zufallszahlen sind mit `random_state = 42` fixiert. PyTorch-Seeds sind gesetzt. Bei identischer Umgebung (`requirements.txt`) sollten alle Ergebnisse bis auf numerische Rundungsdifferenzen exakt reproduzierbar sein.

---

## Kernkonzepte im Detail

### Look-Ahead-Bias

Der zentrale methodische Wendepunkt der Arbeit. In der Voruntersuchung wurde die Standardisierung der Features auf dem Gesamtdatensatz durchgeführt, statt ausschließlich auf den Trainingsdaten. Damit flossen zukünftige Verteilungsinformationen in die Feature-Skalierung ein — ein subtiler, aber schwerwiegender Fehler.

**Korrektur (Notebook 15b):** StandardScaler wird ausschließlich mit `fit_transform(X_train)` auf Trainingsdaten kalibriert, dann per `transform(X_test)` auf Testdaten angewendet. Die zeitliche Trennung wird streng eingehalten.

**Ergebnis der Korrektur:** Die scheinbare Outperformance von +485 % kollabierte zu einer Underperformance von −5,76 pp.

### Bayesian Optimization

Alternative zur Grid Search mit deutlich besserer Sampling-Effizienz. Basierend auf einem Gaussian Process wird die nächste zu testende Hyperparameter-Kombination adaptiv gewählt.

**Verwendete Implementierung:** `scikit-optimize` (`BayesSearchCV`) mit `TimeSeriesSplit(n_splits=5)` als zeitbewusste Cross-Validation.

### Cluster-basierte Basket-Selektion

Zur Vermeidung von Sektor-Konzentration werden die 80 Aktien per K-Means in K=6 Cluster gruppiert (basierend auf Feature-Ähnlichkeit). Aus jedem Cluster wird pro Rebalancing die Aktie mit der höchsten Prognoserendite selektiert. Ergebnis: 6-Aktien-Basket mit struktureller Diversifikation.

---

## Verwendete Bibliotheken

| Kategorie | Bibliothek | Version |
|-----------|------------|---------|
| Datenverarbeitung | pandas | 2.3.3 |
| Numerik | numpy | 2.0.2 |
| Machine Learning | scikit-learn | 1.6.1 |
| Bayesian Optimization | scikit-optimize | 0.10.2 |
| Deep Learning | torch | 2.8.0 |
| Statistik | scipy | 1.13.1 |
| Explainability | shap | 0.49.1 |
| Visualisierung | plotly, matplotlib | 6.9.0 / 3.9 |
| Datenquelle | yfinance | 0.2 |

Vollständige Abhängigkeiten in `requirements.txt`.

---

## Wissenschaftliche Redlichkeit

### Eingesetzte KI-Werkzeuge

Bei dieser Arbeit wurden folgende KI-Werkzeuge eingesetzt:

- **Claude (Sonnet 4.5)** — sprachliche Überarbeitung der Reflexion, methodische Diskussion, Strukturierung der Präsentation
- **Claude Code** — Refactoring von Notebook-Zellen, Debugging bei `scikit-optimize`, Code-Kommentierung
- **Anara** — Literaturrecherche und Zitationsprüfung

Alle inhaltlichen und methodischen Entscheidungen stammen vom Autor. Detaillierte Dokumentation in der beigefügten KI-Dokumentation der Studienarbeit.

---

## Literaturbezug

Die zentralen theoretischen Bezugspunkte:

- **Fama, E. F. (1970).** *Efficient Capital Markets: A Review of Theory and Empirical Work.* Journal of Finance, 25(2), 383-417.
- **López de Prado, M. (2018).** *Advances in Financial Machine Learning.* Wiley — insbesondere Kapitel 11 zu Backtest Overfitting.
- **Snoek, J., Larochelle, H., & Adams, R. P. (2012).** *Practical Bayesian Optimization of Machine Learning Algorithms.* NeurIPS 2012.

---

## Kontakt

**Nassim Groppe**
- GitHub: [@nassimgroppe](https://github.com/nassimgroppe)
- LinkedIn: [Nassim Groppe](https://www.linkedin.com/in/nassim-groppe)

---

## Lizenz

MIT License — siehe [LICENSE](LICENSE)

Die Studienarbeit selbst unterliegt dem Urheberrecht des Autors. Code und Notebooks stehen zur akademischen Nutzung zur Verfügung.

---

<sub>Studienarbeit DBU Data Analyst I · 2026 · Nassim Groppe</sub>
