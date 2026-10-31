# ProMetrics network analysis

Code, data, derived results and vector figures for the manuscript **"Relational leadership and collaboration in the scientific network surrounding digital entrepreneurial marketing"**.

> **Resumen (es).** Este repositorio contiene las exportaciones de Scopus y Web of Science, las tablas generadas por BibFusion (por fuente e integradas), el código que reconstruye la red de coautoría, estima los modelos inferenciales (H1: binomial negativa NB2; H2: GEE logístico) y genera las figuras en formato vectorial. Lea primero *Data protection and licensing notice*.

## What this repository does

1. Builds the coauthorship network from the BibFusion tables (`Article` and `ArticleAuthor`, or `All_Articles` and `All_ArticleAuthor`) and extracts its giant component.
2. Computes, for three windows (2017-2019, 2020-2022, 2023-2025), degree, exact normalized betweenness, recurrent-collaboration share and tenure per author identifier.
3. Estimates the two lagged models of the manuscript:
   - **H1**: new collaborators in the next window; negative binomial (NB2), identifier-clustered covariance.
   - **H2**: persistence in the high-betweenness group; logistic GEE with independent working correlation (Liang-Zeger covariance).
4. Runs robustness checks: Poisson QMLE, any-new-collaborator logit, betweenness x period interaction, cluster bootstrap (B = 200), exclusion of teams with >= 25 authors, and top-5 % / top-20 % thresholds.
5. Writes tables, anonymized network files and figures (SVG, PDF, 600-dpi PNG).

The NB2 and GEE estimators are implemented directly with NumPy/SciPy (no `statsmodels` needed). `--selftest` recovers known parameters from simulated data and checks the analytic gradient; if `statsmodels` is installed, the main run also prints a cross-check against it.

## Repository layout

```
src/         prometrics_network_analysis.py   # pipeline: network, models, figure functions
             make_figures.py                  # figures only (Figs 1-6)
tests/       test_estimators.py               # estimator self-test
data/
  input/                 scopus.csv, wos.txt, scimago.csv.gz     # original exports (see data/README.md)
  bibfusion_output/
    scopus/              final BibFusion tables from the Scopus export  (+ intermediate/)
    wos/                 final BibFusion tables from the WoS export     (+ intermediate/)
    integrated/          All_*.csv: integrated Scopus + WoS tables used by the manuscript
  reference/             country.csv, resumen_tablas.csv
  SHA256SUMS             checksums of every data file
provenance/  configuracion.json, COMPLETADO.json, registro_*.txt, environment_freeze.txt
docs/        data_dictionary.md
results/
  integrated/            main analysis on the integrated tables (tables, figures, network, results.json)
  scopus_replication/    replication on the Scopus-derived tables, reported in the manuscript
```

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sha256sum -c data/SHA256SUMS                                   # optional: verify data integrity
python src/prometrics_network_analysis.py --selftest

# main analysis on the integrated tables (about 2 min; add --boot 0 to skip the bootstrap)
python src/prometrics_network_analysis.py --out results/integrated

# replication on the Scopus-derived tables (reported in the manuscript; about 30 s)
python src/prometrics_network_analysis.py --data data/bibfusion_output/scopus --out results/scopus_replication

# figures only
python src/make_figures.py --out figures
```

Options: `--max-team 25` (large-team threshold), `--boot 200` (bootstrap replicates, `0` to skip), `--seed 42`. `make_figures.py` accepts `--fig1 data` to compute Figure 1 from the tables instead of drawing the numbers reported in the manuscript.

Tested with Python 3.12, NumPy 2.4, pandas 3.0, SciPy 1.17, NetworkX 3.6, Matplotlib 3.10. Figures are drawn at the manuscript's column width (6.575 in) in 10-pt serif type, so they need no rescaling; the font falls back from Times New Roman to Liberation Serif if it is not installed.

## Reproduction check on the integrated tables

Running the pipeline unchanged on `data/bibfusion_output/integrated` gives estimates close to, but not identical with, the principal estimates of the manuscript:

| Quantity | Manuscript | This pipeline |
|---|---|---|
| Identifiers in the giant component | 9,893 | 9,986 |
| Distinct coauthorship links | 65,376 | 65,803 |
| H1 panel observations | 6,681 | 6,725 |
| H1 expected-count ratio (95 % CI) | 1.150 (1.030-1.284) | 1.149 (1.030-1.281) |
| H2 observations / persistence events | 669 / 140 | 683 / 141 |
| H2 odds ratio (95 % CI) | 1.314 (1.108-1.559) | 1.334 (1.124-1.583) |
| Persistence, first / second transition | 78/257 (30.4 %), 62/412 (15.0 %) | 79/269 (29.4 %), 62/414 (15.0 %) |

The small differences come from document linkage. The manuscript's integration matched documents by normalized DOI, then OpenAlex work information, then the SR key, and excluded 370 authorship rows whose metadata were absent or whose match was ambiguous. This repository's loader joins authorships to documents by SR key only and drops the 123 authorship rows without article metadata. Use the manuscript's tables for the published numbers; use this check to confirm that the pipeline implements the same definitions.

## Results included (integrated tables)

`results/integrated/` is the output of `python src/prometrics_network_analysis.py --out results/integrated` (seed 42, 200 bootstrap replicates). Figure 1 in that folder is drawn from the numbers reported in the manuscript; `figS1_composition_replication` is computed from the tables.

| Specification | H1 ratio (95 % CI), p | H2 odds ratio (95 % CI), p |
|---|---|---|
| Principal | 1.149 (1.030-1.281), 0.013 | 1.334 (1.124-1.583), 0.001 |
| Cluster bootstrap CI | 1.033-1.291 | 1.122-1.596 |
| Poisson QMLE | 1.171 (1.084-1.266) | - |
| Excluding documents with >= 25 authors (13 documents, largest 264) | 1.061 (0.991-1.136), 0.088 | 1.287 (1.090-1.521), 0.003 |
| Group threshold top 20 % | - | 1.144 (1.031-1.269), 0.011 |
| Group threshold top 5 % | - | 1.234 (0.968-1.573), 0.090 |

The H2 association is stable across specifications. The H1 association is sensitive to very large teams: it is no longer distinguishable from no association once the 13 documents with 25 or more authors are removed. The NB2 model keeps overdispersion (Pearson 2.93). Confidence intervals are nominal, identifier-clustered and exclude identity uncertainty.

## Results included (Scopus-derived replication)

`results/scopus_replication/` holds the replication reported in the manuscript's "Independent replication and robustness checks" section.

| Association | Estimate | Nominal 95 % CI | p |
|---|---|---|---|
| H1: expected-count ratio per SD of transformed betweenness (NB2) | 1.260 | 1.053-1.507 | 0.012 |
| H2: odds ratio per +10 pp of recurrent collaboration (GEE logit) | 1.236 | 0.899-1.699 | 0.191 |

H1 keeps its direction in every specification but weakens without very large teams (1.154) and under Poisson estimation (1.137; CI 0.986-1.312). H2 is imprecise (42 persistence events in 223 observations) and depends on the threshold defining the group (top 20 %: 1.290; top 5 %: 0.602). See `results/scopus_replication/tables/table_sensitivity.csv` and `results/scopus_replication/figures/`.

## Data protection and licensing notice

- **Personal data.** Several tables contain author names, ORCID iDs, ResearcherIDs and, in `All_Authors.csv`, 354 e-mail addresses, all as they appear in Scopus, Web of Science and OpenAlex records. The raw exports also contain correspondence addresses. Person-level files: `Author*.csv`, `AuthorPerson.csv`, `AuthorAlias.csv`, `AuthorConflicts.csv`, `All_Authors.csv`, the `*author*` intermediate tables, `scopus.csv` and `wos.txt`.
- **Licensing.** Scopus (Elsevier), Web of Science (Clarivate) and SCImago data are subject to their providers' terms. Redistribution rights are the responsibility of the repository owners; see the data statement of the manuscript.
- **Requests.** Authors who want their data corrected or removed can write to [contact e-mail].
- The files written by the code (`results/`) replace people with anonymous numeric codes. Do **not** run with `--export-identities` for anything you intend to publish.

## Limitations

Author identifiers are partly name-based and exposed to homonymy and splitting; repeated `SR` keys may merge distinct documents; coauthorship captures only part of research collaboration; clustering by identifier does not remove dependence among coauthors; no multiplicity adjustment is applied. Results show network positions, not the quality or leadership of any person, and must not be used to rank researchers.

## Citation and license

See `CITATION.cff` (authors and DOI to be completed on release). Code and derived outputs are released under the MIT License (`LICENSE`). The data in `data/` keep the terms of their original providers.

## AI assistance

The replication code, figures and documentation were developed with assistance from Claude (Anthropic) and reviewed by the authors.
