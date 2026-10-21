# Immune Risk Score survival analysis

Transcriptomic DEA + GSEA results feed a simulated Immune Risk Score (IRS) cohort, analysed with Kaplan-Meier, log-rank and Cox PH models.

- `survival_analysis_pipeline.py`: batch pipeline, writes `simulated_cohort.csv` and `survival_analysis_plot.png`
- `app.py`: interactive Streamlit version of the same analysis, plus a viewer for the PPI outputs
- `PPI-Analysis.R`: STRINGdb/igraph network analysis (writes `PPI_*.csv`, `PPI_Network_Plot.png`)

## Introduction

Public expression archives such as The Cancer Genome Atlas (TCGA) have made it routine to search for cancer biomarkers by differential expression. Our group has followed this route for several tumour types. DESeq2 analysis of TCGA lung squamous cell carcinoma data identified CT45A1, GCGR, TPTE, ABCC2 and PI16 as genes whose expression differed between smokers and non-smokers (Shemuel et al., 2023). The same approach applied to colon cancer miRNA profiles gave 107 up-regulated and 189 down-regulated miRNAs (Fugaha et al., 2024). In low-grade glioma, correlation of HspB8 with its candidate regulatory miRNAs pointed to mir-92a-1 as the strongest negative partner (Ivan et al., 2021). The underlying steps are written up as a transcriptomics computational protocol (Parikesit & Agustriawan, 2021), and the wider biomarker landscape is reviewed in Parikesit (2024).

These studies share a limitation that their authors state openly. They stop at differential expression or correlation, and they call for validation in larger independent cohorts. A gene that differs between groups is not yet a prognostic marker. That claim needs a link to clinical outcome, usually time to death or progression, and the statistical tools for it are survival models: the Kaplan-Meier estimator, the log-rank test and Cox proportional-hazards regression. Single genes are also fragile. Integrating several features tends to be more informative, as shown when methylation differences in six Alzheimer's disease genes were combined with a multilayer perceptron classifier, which reached moderate sensitivity and high specificity while still suffering from class imbalance (Mey et al., 2025). Multi-feature scores bring their own risk, though. A recent review of bioinformatics algorithms argues that reporting practice, more than model capacity, now limits reproducibility, and that groups in resource-constrained settings should report enough for others to evaluate their candidates independently (Parikesit, 2026).

This software is a small step from expression results toward outcome. It reads DEA and GSEA tables, collapses the significant immune-related genes into one Immune Risk Score weighted by their fold changes, and runs Kaplan-Meier, log-rank and Cox analyses on that score. No patient outcome data come with the bundled transcriptomic results, so the cohort is simulated from the observed effect sizes. The tool is therefore a method demonstrator and teaching aid. It shows how the score behaves, how sensitive the survival estimates are to each cut-off, and what a full report should contain, before the same code is pointed at a real annotated cohort. Results produced from the simulated cohort are not clinical evidence.

## Implementation

All analysis logic in `app.py` lives in four plain functions, so the Streamlit layer only collects parameters and draws results.

1. **`load_inputs`** reads the DEA table (DESeq2 output with a `symbol` column) and the GSEA table (clusterProfiler `gseGO`-style output). It checks for the required columns and raises a clear error if any are missing, renames `log2FoldChange` to `log2FC`, and coerces GSEA numeric columns.
2. **`simulate_cohort`** follows sections 2 to 5 of `survival_analysis_pipeline.py`, with the same random draws in the same order:
   - Significant DEGs: `padj < cut-off` and `|log2FC| > cut-off`.
   - Immune DEGs: gene symbols starting with one of 41 immune prefixes (HLA, CXCL, IGH, FOXP3, GZMB and others; `CD` and `IL` must be followed by a digit, so CDC6, CDH* and ILK are excluded). With the bundled data this selects 28 genes (CD8A/B, CCL11, IL1RN, IL20RA, immunoglobulin V genes). If none match, all significant DEGs are used.
   - IRS per patient: `IRS_i = sum_j log2FC_j * z_ij`, with `z_ij ~ N(mu_j, 1)`, where `mu_j = -0.15` for down-regulated and `+0.10` for up-regulated genes, standardised to mean 0 and SD 1. Patients are split into High and Low IRS at the median.
   - Survival times: Weibull accelerated failure time model, `log T = log(36) - 0.85 * log(log 2) + beta * IRS + 0.85 * W`, where `W` is a minimum-Gumbel error (so T is Weibull, hazards are proportional, and the median at IRS = 0 is 36 months), with `beta = 0.55 * |mean NES| / 3` taken from the negatively enriched GSEA terms. Higher IRS therefore means longer survival, and the effect size scales with the strength of the GSEA signal.
   - Censoring: administrative at the end of follow-up, plus about 15% random loss to follow-up drawn uniformly between 30% and 100% of the follow-up window. A patient lost to follow-up who died before their censoring time keeps the event.
   - Covariates: age `N(60, 10)` clipped to 20-90 years, and stage 1-4 with probabilities 0.2/0.3/0.3/0.2. Both are drawn independently of survival, so their hazard ratios should sit near 1. This is a useful negative control.
3. **`fit_models`** fits Kaplan-Meier curves per risk group, a two-group log-rank test, a Cox PH model (`IRS + age + stage`, ridge penalizer adjustable), and the Schoenfeld residual test of the proportional-hazards assumption (rank time transform). It uses lifelines 0.30.3.
4. **Plot functions** (`plot_km`, `plot_forest`, `plot_volcano`, `plot_gsea`) return matplotlib figures, which keeps them reusable outside Streamlit.

Input loading is cached with `st.cache_data`, so moving a slider recomputes only the simulation and models, not the file parsing. The PPI tab shows the output of `PPI-Analysis.R` as is. It does not call STRING, so R is not needed to run the app.

## Usage

### Install

Tested with Python 3.11, Streamlit 1.57.0, lifelines 0.30.3 and pandas 2.3.3.

```bash
pip install -r requirements.txt
```

`lifelines` currently requires `pandas < 3`, so installing it may downgrade pandas in a shared environment. Use a dedicated virtual or conda environment if other projects depend on pandas 3.

### Run

```bash
streamlit run app.py      # web app at http://localhost:8501
python app.py             # self-check: default parameters reproduce simulated_cohort.csv
```

Run the self-check after changing any analysis code. It fails if the app no longer matches the batch pipeline.

### Inputs

With no uploads, the app uses `DEA_Full_Results_macOS.csv` and `GSEA_Results_Detailed.csv` from this folder. To analyse other data, upload in the sidebar:

| File | Required columns | Notes |
|---|---|---|
| DEA | first column = gene ID, `log2FoldChange`, `padj`, `symbol` | DESeq2 `results()` with HGNC symbols added |
| GSEA | first column = term ID, `Description`, `NES`, `p.adjust` | clusterProfiler `gseGO`/`gseKEGG` `as.data.frame()` |

### Parameters (sidebar)

| Parameter | Default | Effect |
|---|---|---|
| DEG padj < | 0.05 | Significance cut-off for DEGs entering the IRS |
| \|log2FC\| > | 1.0 | Fold-change cut-off for DEGs |
| GSEA FDR < | 0.05 | Which negatively enriched terms set the simulated effect size |
| Simulated patients | 200 | Cohort size; drives statistical power |
| Follow-up (months) | 60 | Administrative censoring time |
| Random seed | 42 | Change it to see how much results vary between simulated cohorts |
| Cox penalizer | 0.01 | Ridge penalty; raise it if the Cox fit fails to converge |

With the bundled data, the |log2FC| slider has no visible effect below 4, because every gene with padj < 0.05 already has |log2FC| > 3.

### Output tabs

- **Survival**: Kaplan-Meier curves with 95% CI and censor marks, log-rank statistic and p-value, median OS per group, Cox forest plot and table (HR, 95% CI, p), and the PH assumption test. A Schoenfeld p < 0.05 for a covariate suggests its effect changes over time.
- **DEA**: volcano plot at the chosen cut-offs, with the immune genes that make up the IRS listed alongside.
- **GSEA**: lollipop plot and table of negatively enriched terms.
- **PPI network**: precomputed network figure, hub genes and centrality table from `PPI-Analysis.R`.
- **Cohort data**: the simulated patient table, downloadable as CSV.

The header metrics show the number of significant and immune DEGs, events over patients, the IRS hazard ratio per SD with its 95% CI, and the Cox concordance index.

### Reporting a run

Following the reporting point in Parikesit (2026), record at least the input file versions, all sidebar values including the seed, the package versions (`pip freeze`), and the downloaded cohort CSV. Repeat the run with several seeds before reading anything into a single p-value.

### Limitations

- The cohort is simulated. Hazard ratios reflect assumptions built into the simulation, not patient biology.
- Immune gene selection uses gene-symbol prefixes. It misses immune genes with other names and can still admit odd matches for new data. A curated set such as an MSigDB immune signature would be more defensible.
- The median split discards information. The continuous IRS in the Cox model is the better estimate.
- To make prognostic claims, replace `simulate_cohort` with real expression and clinical follow-up data (for example, TCGA clinical tables), keeping `fit_models` unchanged, and validate in an independent cohort.

## References

Fugaha, D. R., Putta, D., Lunoto, D., & Parikesit, A. A. (2024). MicroRNA expression profiling in colorectal carcinoma: Identification and validation of novel diagnostic, prognostic and predictive biomarkers. In *2024 Ninth International Conference on Informatics and Computing (ICIC)* (pp. 1-4). IEEE. https://doi.org/10.1109/ICIC64337.2024.10957168

Ivan, J., Agustriawan, D., Parikesit, A. A., & Nurdiansyah, R. (2021). MiRNA-regulated HspB8 as potent biomarkers in low-grade gliomas. *Research Journal of Biotechnology, 16*(1), 17-25. https://doi.org/10.5281/zenodo.4460727

Mey, T. V. K. Y., Lee, M. B., Tomodok, A. C. A., Muliana, N. E., Priskila, D., Sadrawi, M., & Parikesit, A. A. (2025). Detection of Alzheimer's disease through AI-driven and methylation difference region analysis of significant epigenetic modifications in APP, PSEN1, PSEN2, APOE, MAPT, and TREM2 genes. In A. Isyaku (Ed.), *Advanced therapeutics and disease biology: Molecular diagnostics and immunity - 2025* (pp. 57-86). Halic Publishing House. https://doi.org/10.5281/zenodo.18070507

Parikesit, A. A. (Ed.). (2024). *A closer look at cancer biomarkers*. Nova Science Publishers. https://doi.org/10.52305/wzxv6115

Parikesit, A. A. (2026). Algorithmic perspectives in bioinformatics: A review of computational methods interpreting emerging trends in biotechnology. In B. Akdemir (Ed.), *International Conference on Engineering Sciences* (pp. 669-687). IKSAD. https://ices.kongrepark.org/library

Parikesit, A. A., & Agustriawan, D. (2021). *Transcriptomics computational protocol* [Registered copyright work, guidebook]. i3L; Ditjen HKI. https://e-hakcipta.dgip.go.id/index.php/c?code=MzdiNWM1YzVkYmVlMjM1NThlM2M2ZWJlOWYwMGFlMGIK

Shemuel, J., Angelique, P., Josephine, E., Fugaha, D. R., Gabriela, V., Alhusain, S., & Parikesit, A. A. (2023). Analisis gen komparatif karsinoma sel skuamosa paru-paru antara individu merokok dan tidak merokok [Comparative gene analysis of squamous cell lung carcinoma between smoking and non-smoking individuals]. *Berita Biologi, 22*(3), 291-304. https://doi.org/10.55981/BERITABIOLOGI.2023.908

AI Assistance Disclaimer: This codebase was developed with the assistance of Claude Code. While the AI provided code generation, debugging, and structural support, the human developer maintains full responsibility for reviewing, testing, and maintaining all content and functionality
