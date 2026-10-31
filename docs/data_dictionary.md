# Data dictionary

## Inputs read by the code

`Article.csv` or `All_Articles.csv`: `SR`, `year`, `ismainarticle`. `ArticleAuthor.csv` or `All_ArticleAuthor.csv`: `SR`, `PersonID`. Names are read (only with `--export-identities`) from `AuthorPerson.csv` or `All_Authors.csv`. The remaining BibFusion tables are documented by BibFusion itself.

## Outputs in `<out>/tables/`

| File | Content |
|---|---|
| `table1_network.csv` | Aggregate network: identifiers, links, mean/median degree, density, giant-component share |
| `table2_periods.csv` | Per window: documents, active identifiers, distinct links, positive betweenness, high-betweenness group |
| `table3_main_associations.csv` | H1 and H2 principal estimates (ratio, 95 % CI, p, N) |
| `table_sensitivity.csv` | Large-team exclusion and top-5 % / top-20 % thresholds |
| `table_persistence.csv` | Persistence counts and proportions per transition |
| `model_H1_NB2_full.csv`, `model_H1_NB2_interaction.csv`, `model_H1_PoissonQMLE_full.csv`, `model_H2_GEE_full.csv` | Full coefficient tables (coef, SE, z, p, CI, exponentiated ratio) |
| `panel_identifier_period.csv` | Identifier-period panel; `PersonCode` is an anonymous integer |

### Panel columns
`ndocs` documents in the baseline window; `deg` distinct collaborators; `bc` normalized betweenness; `rec_n`, `rec` collaborators with ≥ 2 shared documents and their share of `deg`; `high` top-decile indicator; `y_new` new collaborators in the next window (H1 outcome); `persist` retained high-betweenness status (H2 outcome; defined only if `high = 1`); `tenure` years since first recorded publication; `trans` transition (1 = 2017–19→2020–22, 2 = 2020–22→2023–25); `tb`, `l_docs`, `l_deg`, `l_ten`, `period2` transformed covariates.

## Network files in `<out>/network/`
`network_nodes.csv` (`id`, `degree`, `n_docs`, `in_main_corpus`, `first_year`), `network_edges.csv` (`u`, `v`, `weight` = shared documents) and `network_giant_component.gexf` (for Gephi). Node ids are anonymous.
