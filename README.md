# What Does Learned Curvature Encode?
## A Structural Analysis of Hyperbolic Knowledge Graph Embeddings

Official code, derived results, statistical analyses, and reproducibility materials for:

> **What Does Learned Curvature Encode? A Structural Analysis of Hyperbolic Knowledge Graph Embeddings**

**Amangel Bhullar** and **Ziad Kobti**  
School of Computer Science  
University of Windsor, Windsor, Ontario, Canada

---

## Overview

Hyperbolic knowledge graph embedding (KGE) methods are commonly motivated by
the observation that hierarchical and tree-like structures can be represented
efficiently in negatively curved spaces. This motivation, however, leaves an
important interpretability question unresolved:

> **What structural properties of a knowledge graph are actually reflected in
> the curvature learned by a hyperbolic embedding model?**

The term *hierarchy* can describe several distinct structural phenomena,
including branching, depth, expansion, directional organization,
tree-likeness, degree heterogeneity, and local graph curvature. These
properties coincide on idealized trees but can vary independently in
real-world knowledge graphs.

This study therefore replaces the broad notion of "hierarchy" with a set of
measurable relation-level structural quantities and examines their association
with learned relation-specific curvature.

The analysis combines:

- six real-world knowledge graph benchmarks;
- relation-level structural characterization;
- multiple independent training seeds;
- rank-based and permutation-based statistical analyses;
- partial correlations controlling for relation frequency;
- false-discovery-rate correction;
- synthetic structural dissociation experiments;
- planted-hierarchy corruption experiments;
- density controls; and
- degree-preserving structural null models.

The goal is not to identify a single universal hierarchy score, but to
determine which aspects of graph structure are reflected in learned
hyperbolic geometry and how those associations change across graph regimes.

---

## Research Questions

The paper investigates the following questions.

### RQ1 — Is learned curvature reproducible?

Do independently trained models assign similar relative curvature to
relations, or are relation-specific curvature estimates dominated by random
initialization and stochastic optimization?

### RQ2 — Which structural properties are associated with learned curvature?

Across real knowledge graphs, how does learned relation-specific curvature
relate to measurable properties such as:

- Forman--Ricci curvature;
- local expansion;
- growth rate;
- branching and fan-out;
- fan-in;
- directional organization;
- hierarchy-related measures;
- tree-likeness;
- cyclic structure;
- degree statistics; and
- relation mapping characteristics?

### RQ3 — Are the observed associations structurally identifiable?

When branching, depth, hierarchy corruption, density, or higher-order
organization are manipulated under controlled conditions, which structural
signals remain associated with curvature?

### RQ4 — Can degree alone explain the observed associations?

Do structure--curvature relationships survive degree-preserving rewiring, or
are they primarily consequences of the degree sequence?

---

## Main Findings

The experiments support several broad conclusions.

### 1. Learned relation-specific curvature is reproducible

Curvature assignments exhibit substantial agreement across independent random
seeds. This indicates that the learned relation-wise geometry contains a
repeatable structural signal rather than being solely an artifact of
stochastic optimization.

The corresponding results are provided in:

```text
results/reproducibility/
```

and summarized visually in:

```text
figures/main/fig1_curvature_reproducibility.pdf
```

### 2. No single structural measure universally explains learned curvature

Structure--curvature associations vary considerably across datasets.

A structural quantity that is strongly associated with curvature in one graph
may be weakly associated, statistically insignificant, or even behave
differently in another graph.

This dataset dependence is a central result of the study and argues against
treating "hierarchy" as a single universal graph property.

Cross-dataset results are available in:

```text
results/cross_dataset/
```

and visualized in:

```text
figures/main/fig2_cross_dataset_heatmap.pdf
```

### 3. Forman--Ricci curvature provides a strong but non-universal signal

Among the investigated structural quantities, Forman--Ricci curvature
provides an important relation-level signal. Its association with learned
curvature, however, is not identical across datasets and should not be
interpreted as a universal explanation of learned geometry.

### 4. Expansion and branching are distinct from depth

The controlled synthetic experiments separate structural factors that are
normally entangled in naturally occurring graphs.

These experiments show that branching, expansion, depth, and cyclic structure
should not be treated as interchangeable definitions of hierarchy.

Results are available under:

```text
results/synthetic_dissociation/
```

with the corresponding figure:

```text
figures/main/fig3_structural_dissociation.pdf
```

### 5. Structural proxies diverge as planted hierarchy is corrupted

In the clean planted-hierarchy regime, several structural measurements can
strongly agree. As non-hierarchical edges are introduced, their behavior
diverges.

For example, the supplied planted-hierarchy experiment evaluates noise levels:

```text
0, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16
```

The results demonstrate that measurements that appear nearly equivalent on an
ideal hierarchy need not remain equivalent after structural corruption.

The complete summary is provided in:

```text
results/planted_hierarchy/summary.json
```

and visualized in:

```text
figures/main/fig4_planted_hierarchy_noise.pdf
```

### 6. Degree is important, but does not provide a complete explanation

Degree-preserving rewiring is used as a structural null model. Because the
rewiring preserves degree-related information while disrupting higher-order
organization, comparing the original and rewired graphs helps distinguish
degree effects from more complex structural effects.

The corresponding results are in:

```text
results/degree_null/
```

and:

```text
figures/main/fig5_degree_preserving_null.pdf
```

---

## Datasets

The cross-dataset analysis uses six knowledge graph benchmarks.

| Dataset | Entities | Relations | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|---:|---:|
| WN18RR | 41,105 | 11 | 86,835 | 3,034 | 3,134 | 93,003 |
| FB15k-237 | 14,541 | 237 | 272,115 | 17,535 | 20,466 | 310,116 |
| YAGO11k | 10,623 | 10 | 16,408 | 2,050 | 2,051 | 20,509 |
| WIKIDATA12k | 12,394 | 24 | 31,224 | 3,986 | 4,006 | 39,216 |
| ICEWS14 | 7,128 | 230 | 63,685 | 13,823 | 13,222 | 90,730 |
| GDELT | 12,869 | 258 | 1,734,399 | 341,961 | 341,961 | 2,418,321 |

The dataset counts recorded during the experiments are available in:

```text
results/datasets/paperA_dataset_counts_final.csv
```

### Dataset redistribution

Raw third-party benchmark datasets are **not redistributed** in this
repository.

The repository instead provides the derived relation-level statistics,
curvature outputs, statistical summaries, and controlled synthetic results
required to inspect the analyses reported in the manuscript.

See:

```text
data/README.md
```

for additional information.

---

## Structural Characterization

Relations are analyzed through their relation-induced subgraphs rather than
through a single global statistic for the entire knowledge graph.

The structural characterization includes several complementary families of
measurements.

### Discrete curvature

Local graph geometry is characterized using discrete curvature measures,
including Forman--Ricci curvature and alternative curvature-related
quantities where applicable.

Relevant implementation:

```text
experiments/recompute_ollivier.py
```

### Expansion and growth

The analysis measures how rapidly relation-induced neighborhoods expand and
whether expansion persists across increasing graph distance.

Relevant implementation:

```text
experiments/volume_growth.py
```

### Branching and directionality

Relation structure is characterized using quantities related to fan-out,
fan-in, head/tail asymmetry, entropy, and directional organization.

### Hierarchical organization

The analysis includes measurements intended to characterize directional and
hierarchical organization without assuming that hierarchy is equivalent to
tree-likeness.

Relevant implementation:

```text
experiments/hierarchy_measures.py
```

### Tree-likeness and cyclic structure

Tree-like organization, cycles, clustering, reciprocity, and related
topological properties are treated separately rather than collapsed into a
single hierarchy variable.

### Degree and relation statistics

Relation frequency, degree-related statistics, and mapping characteristics are
included both as structural descriptors and as potential confounding
variables.

---

## Statistical Analysis

The primary association measure is Spearman rank correlation between learned
relation-specific curvature and relation-level structural quantities.

For each structural covariate, the supplied correlation summaries contain:

```text
covariate
rho_per_seed
rho_pooled
rho_mean_of_seeds
ci95
perm_p
partial_rho_ctrl_logfreq
```

These provide:

- correlation estimates for individual random seeds;
- pooled Spearman correlation;
- mean correlation across seeds;
- confidence intervals;
- permutation-based significance tests; and
- partial correlations controlling for log relation frequency.

### Multiple-comparison correction

Benjamini--Hochberg false-discovery-rate correction is used when evaluating
multiple structural covariates.

The corresponding outputs are available under:

```text
results/statistics/fdr_all_covariate_results.csv
results/statistics/fdr_counts_by_dataset.csv
results/statistics/fdr_results_full.csv
```

Relevant implementation:

```text
experiments/fdr_all_covariates.py
```

### Partial correlation and partial permutation analysis

Frequency-controlled analyses are provided in:

```text
results/statistics/partial_correlations.csv
results/statistics/partial_permutation_summary.csv
results/statistics/partial_permutation_summary_v2.csv
```

Relevant scripts are:

```text
experiments/partial_perm_test.py
experiments/partial_perm_test_v2.py
```

---

## Controlled Structural Experiments

Observational correlations alone cannot establish which structural property is
responsible for a learned geometric signal. The repository therefore contains
several controlled experiments.

### Synthetic structural dissociation

Synthetic relation families vary structural properties such as branching,
depth, and cross-structure connectivity.

The relation profiles and resulting statistics are stored in:

```text
results/synthetic_dissociation/
```

Synthetic graph generation:

```text
experiments/make_synthetic.py
```

### Controlled growth regimes

A second synthetic benchmark contains structural regimes including trees,
stars, chains, grids, and cliques.

The supplied relation profiles include examples such as:

```text
tree_b2_d6
tree_b3_d4
tree_b4_d3
star_b8
star_b32
chain_d8
chain_d16
grid_8
grid_12
clique_6
clique_10
```

Results are available in:

```text
results/growth_regimes/
```

Generation code:

```text
experiments/make_synthetic_trees.py
```

### Planted hierarchy and noise

A planted hierarchical graph is progressively corrupted with additional
non-hierarchical structure.

Implementation:

```text
experiments/planted_hierarchy.py
```

Results:

```text
results/planted_hierarchy/summary.json
```

### Density controls

Density is varied independently to examine whether apparent structural
associations can arise from density changes or loss of structural variance.

Implementation:

```text
experiments/density_sweep.py
```

Results:

```text
results/density/summary.json
```

### Degree-preserving structural nulls

Degree-preserving rewiring disrupts higher-order graph organization while
retaining degree-related constraints.

Implementation:

```text
experiments/rewire_null.py
```

Results:

```text
results/degree_null/
```

---

## Repository Structure

```text
learned-curvature-structure/
│
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md
│
├── experiments/
│   ├── analyze_correlation.py
│   ├── density_sweep.py
│   ├── fdr_all_covariates.py
│   ├── hierarchy_measures.py
│   ├── make_paperA_figures_final.py
│   ├── make_synthetic.py
│   ├── make_synthetic_trees.py
│   ├── partial_perm_test.py
│   ├── partial_perm_test_v2.py
│   ├── planted_hierarchy.py
│   ├── recompute_ollivier.py
│   ├── rewire_null.py
│   └── volume_growth.py
│
├── figures/
│   ├── main/
│   │   ├── fig1_curvature_reproducibility.pdf
│   │   ├── fig2_cross_dataset_heatmap.pdf
│   │   ├── fig3_structural_dissociation.pdf
│   │   ├── fig4_planted_hierarchy_noise.pdf
│   │   └── fig5_degree_preserving_null.pdf
│   │
│   └── appendix/
│       ├── figA1_density_correlations.pdf
│       ├── figA2_density_variance_collapse.pdf
│       └── figA3_growth_regimes.pdf
│
└── results/
    ├── cross_dataset/
    ├── datasets/
    ├── degree_null/
    ├── density/
    ├── growth_regimes/
    ├── planted_hierarchy/
    ├── reproducibility/
    ├── statistics/
    └── synthetic_dissociation/
```

---

## Results Directory

### `results/cross_dataset/`

Contains the final relation-level structural statistics and
structure--curvature correlation summaries for:

```text
WN18RR
FB15k-237
YAGO11k
WIKIDATA12k
ICEWS14
GDELT
```

For each dataset, the repository includes:

```text
*_relation_stats.csv
*_correlation_summary.csv
```

### `results/reproducibility/`

Contains learned relation-specific curvature values from three independent
seeds for each of the six empirical datasets.

These files support the cross-seed reproducibility analysis.

### `results/statistics/`

Contains the final inferential statistical analyses, including FDR-corrected
results and partial permutation analyses.

### `results/synthetic_dissociation/`

Contains relation profiles, structural statistics, correlation summaries, and
five independent curvature runs for the synthetic structural dissociation
experiment.

### `results/growth_regimes/`

Contains relation profiles and results for controlled tree, star, chain, grid,
and clique regimes.

### `results/planted_hierarchy/`

Contains the complete planted-hierarchy noise sweep.

### `results/density/`

Contains the density-control experiment summary.

### `results/degree_null/`

Contains original and degree-preserving-rewired results for WN18RR and
YAGO11k.

---

## Figures

### Main-paper figures

| Figure | File | Purpose |
|---|---|---|
| Fig. 1 | `fig1_curvature_reproducibility.pdf` | Cross-seed reproducibility of learned curvature |
| Fig. 2 | `fig2_cross_dataset_heatmap.pdf` | Cross-dataset structure--curvature associations |
| Fig. 3 | `fig3_structural_dissociation.pdf` | Controlled structural dissociation |
| Fig. 4 | `fig4_planted_hierarchy_noise.pdf` | Planted hierarchy under increasing corruption |
| Fig. 5 | `fig5_degree_preserving_null.pdf` | Degree-preserving structural null analysis |

### Appendix figures

| Figure | File | Purpose |
|---|---|---|
| Fig. A1 | `figA1_density_correlations.pdf` | Density-control correlations |
| Fig. A2 | `figA2_density_variance_collapse.pdf` | Structural variance under density changes |
| Fig. A3 | `figA3_growth_regimes.pdf` | Controlled growth regimes |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/amangelbhullar/learned-curvature-structure.git
cd learned-curvature-structure
```

A Python virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

The current dependency list includes the main scientific Python packages used
by the analysis:

```text
numpy
pandas
scipy
matplotlib
networkx
torch
scikit-learn
statsmodels
```

Exact package versions may depend on the environment used to rerun individual
experiments.

---

## Reproducing the Analysis

The repository is organized so that the supplied derived outputs can be
inspected independently of the raw third-party datasets.

### Cross-dataset correlation analysis

The main correlation analysis is implemented in:

```bash
python experiments/analyze_correlation.py
```

The final supplied outputs are available in:

```text
results/cross_dataset/
```

### Structural measurements

Hierarchy-related measurements:

```bash
python experiments/hierarchy_measures.py
```

Volume-growth measurements:

```bash
python experiments/volume_growth.py
```

Alternative discrete-curvature calculations:

```bash
python experiments/recompute_ollivier.py
```

These scripts may require paths to the corresponding benchmark data depending
on the experiment being reproduced.

### Statistical correction

```bash
python experiments/fdr_all_covariates.py
```

### Partial permutation analysis

```bash
python experiments/partial_perm_test_v2.py
```

### Controlled experiments

Synthetic dissociation:

```bash
python experiments/make_synthetic.py
```

Controlled structural regimes:

```bash
python experiments/make_synthetic_trees.py
```

Planted hierarchy:

```bash
python experiments/planted_hierarchy.py
```

Density sweep:

```bash
python experiments/density_sweep.py
```

Degree-preserving rewiring:

```bash
python experiments/rewire_null.py
```

---

## Reproducing the Figures

The final figure-generation script is:

```text
experiments/make_paperA_figures_final.py
```

Run:

```bash
python experiments/make_paperA_figures_final.py
```

The repository already contains the derived result files used to construct the
paper figures.

Final publication-quality PDF versions are included under:

```text
figures/main/
figures/appendix/
```

---

## Reproducibility Notes

Several design decisions are important when interpreting or reproducing the
experiments.

### Relation-level analysis

The unit of structural analysis is the relation-induced subgraph. Structural
statistics are therefore computed separately for relations rather than only
once for the complete knowledge graph.

### Multiple random seeds

Learned curvature is evaluated across multiple independent runs so that
structure--curvature associations are not inferred from a single model
initialization.

### Rank-based association

Spearman correlation is used because the analysis primarily concerns
monotonic relation-level association and does not assume a linear relationship
between structural measurements and learned curvature.

### Relation frequency

Relation frequency can influence both learned parameters and structural
measurements. Partial analyses therefore control for log relation frequency.

### Multiple testing

Because many structural covariates are examined, false-discovery-rate
correction is included to distinguish nominal significance from associations
that survive multiple-comparison control.

### Controlled interventions

Synthetic experiments are used to separate structural variables that are
strongly correlated in naturally occurring graphs.

### Structural null models

Degree-preserving rewiring provides a stronger control than comparison against
an unconstrained random graph because it retains degree-related information
while disrupting higher-order organization.

---

## Scope of the Repository

This repository is intended to support the structural analysis presented in
the associated manuscript.

It contains:

- structural-analysis code;
- controlled graph-generation code;
- relation-level structural statistics;
- learned curvature outputs used by the analysis;
- correlation summaries;
- permutation-test results;
- partial-correlation results;
- FDR-corrected statistical results;
- controlled-experiment summaries;
- degree-preserving-null results; and
- final manuscript figures.

It does **not** redistribute:

- raw third-party benchmark datasets;
- large model checkpoints; or
- unrelated experiments from other research projects.

This separation keeps the repository focused on the experiments reported in
the paper.

---

## Limitations

The analyses in this repository should not be interpreted as demonstrating
that one structural quantity universally determines hyperbolic curvature.

The empirical results instead indicate that learned curvature reflects a
combination of structural properties whose relative importance can depend on
the graph, relation distribution, density, and structural regime.

Furthermore, observational correlations in real datasets do not by themselves
establish causality. The controlled synthetic experiments and structural null
models are included specifically to strengthen structural interpretation.

---

## Citation

If you use the code, results, or experimental methodology from this repository,
please cite the associated paper:

```bibtex
@article{bhullar2026learnedcurvature,
  title   = {What Does Learned Curvature Encode? A Structural Analysis of
             Hyperbolic Knowledge Graph Embeddings},
  author  = {Bhullar, Amangel and Kobti, Ziad},
  year    = {2026},
  note    = {Manuscript}
}
```

The citation information will be updated with the journal, volume, pages, and
DOI after publication.

Machine-readable citation metadata is also provided in:

```text
CITATION.cff
```

---

## Authors

### Amangel Bhullar
School of Computer Science  
University of Windsor  
Windsor, Ontario, Canada

### Ziad Kobti
School of Computer Science  
University of Windsor  
Windsor, Ontario, Canada

---

## License

The research code in this repository is released under the terms specified in
the `LICENSE` file.

Third-party datasets remain subject to their respective licenses and terms of
use.

---

## Repository

**GitHub:**  
https://github.com/amangelbhullar/learned-curvature-structure
