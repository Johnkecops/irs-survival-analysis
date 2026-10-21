#!/usr/bin/env python3
"""
=============================================================================
 SURVIVAL ANALYSIS PIPELINE
 Transcriptomic data: DEA + GSEA → Immune Risk Score → Survival Modelling
=============================================================================
 Author  : Pipeline for Dr. Arli Aditya Parikesit, i3L (ORCID: 0000-0001-8716-3926)
 Date    : 2026-04-16
 Inputs  : DEA_results_3_samples.csv
           DEA_Full_Results_macOS.csv
           GSEA_Results_Detailed.csv

 Biological rationale
 --------------------
 GSEA identifies robust negative enrichment of immune response pathways
 (top terms NES ≈ −2.5 to −3.1, FDR < 1e-8): lymphocyte-mediated immunity,
 B-cell / T-cell signalling, and antigen-receptor pathways.  Immune
 exclusion / suppression in the tumour micro-environment (TME) is a
 well-replicated poor-prognosis marker across haematological and solid
 cancers (Galon & Bruni, Nat Rev Cancer 2019).

 Analytical strategy
 -------------------
 1. Extract top significant DEGs (padj < 0.05, |log2FC| > 1).
 2. Identify immune-related DEGs by keyword matching gene symbols.
 3. Build a composite Immune Risk Score (IRS) per simulated patient:
      IRS_i = Σ_j [ log2FC_j × z_ij ]
    where z_ij ~ N(0,1) represents patient-specific expression deviation
    for gene j.  Positive IRS → high immune activity (better prognosis);
    negative IRS → immune suppression (poor prognosis), consistent with
    the observed GSEA signal.
 4. Stratify patients into High / Low IRS groups (median split).
 5. Kaplan–Meier estimator + log-rank test.
 6. Cox proportional-hazards regression with continuous IRS and
    additional covariates (age, stage).
 7. Publication-quality multi-panel figure.

=============================================================================
"""

import os, re, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from scipy import stats
warnings.filterwarnings('ignore', category=FutureWarning)  # keep lifelines convergence/PH warnings visible

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Paths & global config
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = BASE   # save figures alongside the script

np.random.seed(42)
N_PATIENTS       = 200   # simulated cohort
FOLLOW_UP_MONTHS = 60    # 5-year observation window
ALPHA            = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load data
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  SURVIVAL ANALYSIS PIPELINE  —  TRANSCRIPTOMIC DATA")
print("=" * 70)

dea_3    = pd.read_csv(os.path.join(BASE, "DEA_results_3_samples.csv"),   index_col=0)
dea_full = pd.read_csv(os.path.join(BASE, "DEA_Full_Results_macOS.csv"),  index_col=0)
gsea     = pd.read_csv(os.path.join(BASE, "GSEA_Results_Detailed.csv"),   index_col=0)

# Normalise column names
dea_3.columns    = ['baseMean','log2FC','lfcSE','stat','pvalue','padj']
dea_full.columns = ['baseMean','log2FC','lfcSE','stat','pvalue','padj','symbol']

print(f"\n[Data loaded]")
print(f"  DEA 3-sample  : {dea_3.shape[0]:,} genes")
print(f"  DEA full      : {dea_full.shape[0]:,} genes  |  {dea_full['symbol'].notna().sum():,} with symbols")
print(f"  GSEA pathways : {gsea.shape[0]:,} terms")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  DEA summary statistics
# ─────────────────────────────────────────────────────────────────────────────
# Significant DEGs: padj < 0.05  AND  |log2FC| > 1
sig_full = dea_full[(dea_full['padj'] < 0.05) &
                    (dea_full['log2FC'].abs() > 1)].copy()

sig_3 = dea_3[(dea_3['padj'] < 0.05) &
              (dea_3['log2FC'].abs() > 1)].copy()

up_full   = sig_full[sig_full['log2FC'] > 0]
down_full = sig_full[sig_full['log2FC'] < 0]

print(f"\n[Significant DEGs — Full dataset  (padj<0.05, |log2FC|>1)]")
print(f"  Up-regulated   : {len(up_full):>5,}")
print(f"  Down-regulated : {len(down_full):>5,}")
print(f"  Total          : {len(sig_full):>5,}")

print(f"\n[Significant DEGs — 3-sample set  (padj<0.05, |log2FC|>1)]")
print(f"  Total          : {len(sig_3):>5,}")
print(f"  Top-5 by |log2FC|:")
top5 = sig_3.loc[sig_3['log2FC'].abs().nlargest(5).index, ['baseMean','log2FC','padj']]
print(top5.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3.  GSEA summary
# ─────────────────────────────────────────────────────────────────────────────
# Ensure numeric columns
for col in ['NES','pvalue','p.adjust','setSize']:
    gsea[col] = pd.to_numeric(gsea[col], errors='coerce')

sig_gsea = gsea[gsea['p.adjust'] < 0.05].copy()
neg_gsea = sig_gsea[sig_gsea['NES'] < 0]
pos_gsea = sig_gsea[sig_gsea['NES'] > 0]

print(f"\n[GSEA  (FDR<0.05)]")
print(f"  Negatively enriched (immune suppression) : {len(neg_gsea):>4,} pathways")
print(f"  Positively enriched                      : {len(pos_gsea):>4,} pathways")
print(f"\n  Top-10 negatively enriched pathways (|NES|):")
top_neg = neg_gsea.nsmallest(10, 'NES')[['Description','NES','p.adjust','setSize']]
print(top_neg.to_string())

# Mean NES of immune pathways (used later to scale hazard ratio)
mean_neg_nes = neg_gsea['NES'].mean()
print(f"\n  Mean NES (negative terms): {mean_neg_nes:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Immune-related DEGs — used to construct IRS
# ─────────────────────────────────────────────────────────────────────────────
IMMUNE_KEYWORDS = [
    'CD', 'HLA', 'IL', 'IFN', 'TNF', 'FCR', 'MHC', 'TLR', 'CXCL', 'CCL',
    'CXCR', 'CCR', 'IGH', 'IGL', 'IGK', 'TCR', 'TRAC', 'TRBC', 'PDCD',
    'CTLA', 'LAG', 'TIGIT', 'HAVCR', 'TNFRSF', 'TNFSF', 'NCR', 'NKG',
    'KLRD', 'KLRC', 'KLRB', 'BTK', 'SYK', 'LCK', 'ZAP70', 'PTPRC',
    'ITGAM', 'ITGAX', 'FOXP3', 'GZMB', 'PRF1', 'EOMES', 'TBX21'
]

# Prefix match; CD/IL need a digit so CDC6, CDH*, ILK are excluded.
# ponytail: keyword prefixes still approximate; swap for a curated immune gene set (e.g. MSigDB) if precision matters
IMMUNE_RE = re.compile('|'.join(k + r'\d' if k in ('CD', 'IL') else k for k in IMMUNE_KEYWORDS))

def is_immune(symbol):
    return pd.notna(symbol) and bool(IMMUNE_RE.match(str(symbol).upper()))

sig_full['immune_flag'] = sig_full['symbol'].apply(is_immune)
immune_degs = sig_full[sig_full['immune_flag']].copy()

print(f"\n[Immune-related DEGs]")
print(f"  Found {len(immune_degs)} immune-annotated significant DEGs")
if len(immune_degs) == 0:
    # Fall back to ALL significant DEGs if no immune match
    print("  ⚠  No immune-keyword matches — using all significant DEGs for IRS")
    immune_degs = sig_full.copy()

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Simulate patient cohort
# ─────────────────────────────────────────────────────────────────────────────
# For each patient i, gene j gets a 'relative expression' z_ij ~ N(0,1).
# IRS_i  = Σ_j  log2FC_j * z_ij  (summed over immune DEGs)
#
# GSEA shows the bulk sample is IMMUNE-SUPPRESSED (negative NES).
# We therefore skew the IRS distribution slightly negative to reflect this:
#   z_ij ~ N(mu_skew, 1)  where mu_skew < 0 for down-regulated genes
#          (i.e., down-regulated immune genes are expressed even lower
#           in the tumour than in normal tissue on average).

weights = immune_degs['log2FC'].values
n_genes = len(weights)

# Each patient draws a deviation vector; suppress direction is encoded
# by mixing in a small negative mean for genes with log2FC < 0
patient_z = np.zeros((N_PATIENTS, n_genes))
for g, w in enumerate(weights):
    mu_g = -0.15 if w < 0 else 0.10   # slight bias consistent with GSEA
    patient_z[:, g] = np.random.normal(mu_g, 1.0, N_PATIENTS)

IRS = patient_z @ weights   # shape (N_patients,)

# Standardise
IRS_std = (IRS - IRS.mean()) / IRS.std()

# Risk group: High (IRS ≥ median) vs Low (IRS < median)
median_irs = np.median(IRS_std)
risk_group = np.where(IRS_std >= median_irs, "High", "Low")

# ── Survival time simulation ──────────────────────────────────────────────
# Weibull AFT model:
#   log(T_i) = μ + β * IRS_std_i + σ * ε_i,  ε_i ~ Gumbel(0,1)
#
# Parameters chosen so that:
#   • overall median survival  ≈ 36 months
#   • High-IRS HR ≈ 0.50 relative to Low-IRS (better immune → better prognosis)
#   Hazard ratio calibrated from |mean_neg_nes| scaled to realistic range.
#
mu_aft   = np.log(36) - 0.85 * np.log(np.log(2))  # intercept → median = 36 mo under min-Gumbel errors
hr_scale = abs(mean_neg_nes) / 3.0  # maps NES to log-HR (≈ 0.66 for bundled GSEA, mean NES ≈ −1.99)
beta_irt = 0.55 * hr_scale       # protective effect of high IRS
sigma_aft = 0.85                 # shape

eps   = -np.random.gumbel(0, 1, N_PATIENTS)   # min-Gumbel → Weibull T (PH holds)
log_T = mu_aft + beta_irt * IRS_std + sigma_aft * eps
T_true = np.exp(log_T)

# Administrative censoring at FOLLOW_UP_MONTHS
# + random loss-to-follow-up: ~15 % censored uniformly
cens_time = np.random.uniform(FOLLOW_UP_MONTHS * 0.3, FOLLOW_UP_MONTHS, N_PATIENTS)
mask_ltfu  = np.random.binomial(1, 0.15, N_PATIENTS).astype(bool)

T_obs   = np.minimum(T_true, FOLLOW_UP_MONTHS)
event   = (T_true <= FOLLOW_UP_MONTHS).astype(int)
T_obs[mask_ltfu] = np.minimum(cens_time[mask_ltfu], T_obs[mask_ltfu])
event[mask_ltfu] = (T_true[mask_ltfu] <= T_obs[mask_ltfu]).astype(int)   # death before LTFU censor stays an event

# Covariates
age   = np.random.normal(60, 10, N_PATIENTS).clip(20, 90)
stage = np.random.choice([1, 2, 3, 4],
                         p=[0.20, 0.30, 0.30, 0.20],
                         size=N_PATIENTS)

# Build cohort dataframe
cohort = pd.DataFrame({
    'patient_id'  : [f"PT{i:04d}" for i in range(N_PATIENTS)],
    'IRS'         : IRS_std,
    'risk_group'  : risk_group,
    'time_months' : T_obs,
    'event'       : event,
    'age'         : age,
    'stage'       : stage,
})
cohort.to_csv(os.path.join(OUTDIR, "simulated_cohort.csv"), index=False)

print(f"\n[Patient cohort (n={N_PATIENTS})]")
print(f"  Events (deaths/progressions) : {event.sum()} / {N_PATIENTS}  ({100*event.mean():.1f} %)")
print(f"  Median follow-up             : {np.median(T_obs):.1f} months")
print(f"  High-IRS patients            : {(risk_group=='High').sum()}")
print(f"  Low-IRS  patients            : {(risk_group=='Low').sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Kaplan–Meier estimation
# ─────────────────────────────────────────────────────────────────────────────
high_mask = cohort['risk_group'] == 'High'
low_mask  = cohort['risk_group'] == 'Low'

kmf_high = KaplanMeierFitter()
kmf_low  = KaplanMeierFitter()
kmf_all  = KaplanMeierFitter()

kmf_high.fit(cohort.loc[high_mask, 'time_months'],
             cohort.loc[high_mask, 'event'],
             label='High IRS (immune-active)')
kmf_low.fit( cohort.loc[low_mask,  'time_months'],
             cohort.loc[low_mask,  'event'],
             label='Low IRS (immune-suppressed)')
kmf_all.fit( cohort['time_months'], cohort['event'], label='Overall')

# ── Log-rank test ──────────────────────────────────────────────────────────
lr = logrank_test(
    cohort.loc[high_mask, 'time_months'], cohort.loc[low_mask, 'time_months'],
    cohort.loc[high_mask, 'event'],       cohort.loc[low_mask, 'event'],
)
print(f"\n[Log-rank test  High vs Low IRS]")
print(f"  Test statistic : {lr.test_statistic:.4f}")
print(f"  p-value        : {lr.p_value:.2e}")
print(f"  Significant    : {'YES' if lr.p_value < ALPHA else 'NO'}  (α={ALPHA})")

# Median survival
med_high = kmf_high.median_survival_time_
med_low  = kmf_low.median_survival_time_
print(f"\n  Median OS — High IRS : {med_high:.1f} months")
print(f"  Median OS — Low  IRS : {med_low:.1f} months")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Cox proportional-hazards regression
# ─────────────────────────────────────────────────────────────────────────────
cox_df = cohort[['time_months','event','IRS','age','stage']].copy()

cph = CoxPHFitter(penalizer=0.01)
cph.fit(cox_df, duration_col='time_months', event_col='event')

print(f"\n[Cox PH — multivariate model]")
cph.print_summary()

# Extract key stats
cox_summary = cph.summary.copy()
irs_hr   = np.exp(cph.params_['IRS'])
irs_ci   = np.exp(cph.confidence_intervals_.loc['IRS'].values)
irs_pval = cox_summary.loc['IRS', 'p']

print(f"\n  IRS hazard ratio  : {irs_hr:.3f}  (95 % CI: {irs_ci[0]:.3f}–{irs_ci[1]:.3f})")
print(f"  IRS p-value       : {irs_pval:.2e}")
print(f"  Concordance index : {cph.concordance_index_:.4f}")

# Proportional-hazards assumption test
from lifelines.statistics import proportional_hazard_test
phtest = proportional_hazard_test(cph, cox_df, time_transform='rank')
print(f"\n[Schoenfeld residuals — PH assumption]")
print(phtest.summary.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Nelson–Aalen cumulative hazard
# ─────────────────────────────────────────────────────────────────────────────
naf_high = NelsonAalenFitter()
naf_low  = NelsonAalenFitter()
naf_high.fit(cohort.loc[high_mask,'time_months'], cohort.loc[high_mask,'event'],
             label='High IRS')
naf_low.fit( cohort.loc[low_mask, 'time_months'], cohort.loc[low_mask, 'event'],
             label='Low IRS')

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Volcano plot data prep (for DEA panel)
# ─────────────────────────────────────────────────────────────────────────────
volcano_df = dea_full.dropna(subset=['padj','log2FC']).copy()
volcano_df['neglog10p'] = -np.log10(volcano_df['padj'].clip(lower=1e-300))
volcano_df['sig'] = 'NS'
volcano_df.loc[(volcano_df['padj'] < 0.05) & (volcano_df['log2FC'] >  1), 'sig'] = 'Up'
volcano_df.loc[(volcano_df['padj'] < 0.05) & (volcano_df['log2FC'] < -1), 'sig'] = 'Down'

# Top-label genes
top_label = volcano_df[volcano_df['sig'] != 'NS'].nlargest(10, 'neglog10p')

# ─────────────────────────────────────────────────────────────────────────────
# 10.  GSEA lollipop data
# ─────────────────────────────────────────────────────────────────────────────
gsea_plot = (gsea[gsea['p.adjust'] < 0.05]
             .nsmallest(15, 'NES')[['Description','NES','p.adjust','setSize']]
             .sort_values('NES'))

# Truncate long names
gsea_plot['short_name'] = gsea_plot['Description'].str[:50]

# ─────────────────────────────────────────────────────────────────────────────
# 11.  FIGURE — 6-panel publication layout
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {'High IRS (immune-active)' : '#2166ac',
           'Low IRS (immune-suppressed)': '#d6604d'}
HIGH_COL = '#2166ac'
LOW_COL  = '#d6604d'

fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor('#FAFAFA')
gs  = gridspec.GridSpec(3, 3, figure=fig,
                        hspace=0.42, wspace=0.38,
                        top=0.93, bottom=0.05,
                        left=0.07, right=0.97)

ax_km   = fig.add_subplot(gs[0, :2])   # Kaplan–Meier  (spans 2 cols)
ax_na   = fig.add_subplot(gs[0, 2])    # Nelson–Aalen
ax_vol  = fig.add_subplot(gs[1, 0])    # Volcano
ax_gsea = fig.add_subplot(gs[1, 1:])   # GSEA lollipop (spans 2 cols)
ax_irs  = fig.add_subplot(gs[2, 0])    # IRS distribution
ax_cox  = fig.add_subplot(gs[2, 1:])   # Cox forest plot

# ── SUPTITLE ─────────────────────────────────────────────────────────────────
fig.suptitle(
    "Survival Analysis  ·  Immune Risk Score derived from DEA & GSEA",
    fontsize=16, fontweight='bold', color='#1a1a2e', y=0.97
)

# ─── Panel A  Kaplan–Meier ────────────────────────────────────────────────────
kmf_high.plot_survival_function(
    ax=ax_km, ci_show=True, color=HIGH_COL, linewidth=2.5)
kmf_low.plot_survival_function(
    ax=ax_km, ci_show=True, color=LOW_COL, linewidth=2.5)

# At-risk table
from lifelines.plotting import add_at_risk_counts
add_at_risk_counts(kmf_high, kmf_low, ax=ax_km,
                   fontsize=9, xticks=[0,12,24,36,48,60])

ax_km.set_xlabel("Time (months)", fontsize=11)
ax_km.set_ylabel("Survival probability", fontsize=11)
ax_km.set_title("A  Kaplan–Meier Overall Survival", fontweight='bold', fontsize=12, loc='left')
ax_km.set_xlim(0, FOLLOW_UP_MONTHS)
ax_km.set_ylim(0, 1.05)

# Log-rank annotation
pstr = f"p = {lr.p_value:.2e}" if lr.p_value >= 1e-300 else "p < 1e-300"
sig_mark = "***" if lr.p_value < 0.001 else "**" if lr.p_value < 0.01 else "*" if lr.p_value < ALPHA else "ns"
ax_km.annotate(f"Log-rank {sig_mark}\n{pstr}",
               xy=(0.98, 0.72), xycoords='axes fraction',
               ha='right', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#aaa', alpha=0.9))

# Median lines
for med, col in [(med_high, HIGH_COL), (med_low, LOW_COL)]:
    if not np.isinf(med) and not np.isnan(med):
        ax_km.axvline(med, color=col, linestyle='--', alpha=0.6, linewidth=1.2)
        ax_km.text(med + 0.5, 0.52, f'{med:.0f} mo',
                   color=col, fontsize=8.5, va='bottom')

ax_km.axhline(0.5, color='grey', linestyle=':', linewidth=1, alpha=0.5)
ax_km.legend(loc='upper right', fontsize=9.5, framealpha=0.9)
ax_km.spines[['top','right']].set_visible(False)

# ─── Panel B  Nelson–Aalen cumulative hazard ─────────────────────────────────
naf_high.plot_cumulative_hazard(ax=ax_na, color=HIGH_COL, linewidth=2.2,
                                 ci_show=True, label='High IRS')
naf_low.plot_cumulative_hazard( ax=ax_na, color=LOW_COL,  linewidth=2.2,
                                 ci_show=True, label='Low IRS')
ax_na.set_xlabel("Time (months)", fontsize=10)
ax_na.set_ylabel("Cumulative hazard H(t)", fontsize=10)
ax_na.set_title("B  Nelson–Aalen Cumulative Hazard", fontweight='bold', fontsize=11, loc='left')
ax_na.set_xlim(0, FOLLOW_UP_MONTHS)
ax_na.legend(fontsize=9, framealpha=0.9)
ax_na.spines[['top','right']].set_visible(False)

# ─── Panel C  Volcano plot ────────────────────────────────────────────────────
color_map = {'NS': '#CCCCCC', 'Up': '#c0392b', 'Down': '#2980b9'}
for cat, grp in volcano_df.groupby('sig'):
    ax_vol.scatter(grp['log2FC'], grp['neglog10p'],
                   c=color_map[cat], s=8, alpha=0.55,
                   label=cat, rasterized=True)

ax_vol.axvline( 1, color='grey', linestyle='--', linewidth=0.9, alpha=0.7)
ax_vol.axvline(-1, color='grey', linestyle='--', linewidth=0.9, alpha=0.7)
ax_vol.axhline(-np.log10(0.05), color='grey', linestyle=':', linewidth=0.9, alpha=0.7)

# Label top genes
for _, row in top_label.head(6).iterrows():
    sym = str(row['symbol']) if not pd.isna(row['symbol']) else ''
    ax_vol.annotate(sym,
                    (row['log2FC'], row['neglog10p']),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=6.5, color='#333333',
                    arrowprops=dict(arrowstyle='-', lw=0.5, color='#999'))

ax_vol.set_xlabel("log₂ Fold Change", fontsize=10)
ax_vol.set_ylabel("-log₁₀(padj)", fontsize=10)
ax_vol.set_title("C  Volcano Plot — DEA Full", fontweight='bold', fontsize=11, loc='left')
leg_handles = [Patch(facecolor=c, label=l)
               for l, c in [('Up', '#c0392b'),('Down','#2980b9'),('NS','#CCCCCC')]]
ax_vol.legend(handles=leg_handles, fontsize=8.5, loc='upper left', framealpha=0.9)
ax_vol.spines[['top','right']].set_visible(False)

# ─── Panel D  GSEA lollipop ──────────────────────────────────────────────────
bar_colors = [LOW_COL if n < 0 else HIGH_COL for n in gsea_plot['NES']]
y_pos = range(len(gsea_plot))

ax_gsea.barh(list(y_pos), gsea_plot['NES'].values,
             color=bar_colors, alpha=0.80, height=0.65)
ax_gsea.scatter(gsea_plot['NES'].values, list(y_pos),
                color=bar_colors, zorder=5, s=50, edgecolors='white', linewidths=0.5)
ax_gsea.axvline(0, color='black', linewidth=0.8)

# FDR annotation
for i, (_, row) in enumerate(gsea_plot.iterrows()):
    fdr_str = f"FDR={row['p.adjust']:.1e}"
    x_off = row['NES'] - 0.08 if row['NES'] < 0 else row['NES'] + 0.04
    ax_gsea.text(x_off, i, fdr_str, va='center', fontsize=7, color='#444')

ax_gsea.set_yticks(list(y_pos))
ax_gsea.set_yticklabels(gsea_plot['short_name'].values, fontsize=8.5)
ax_gsea.set_xlabel("Normalised Enrichment Score (NES)", fontsize=10)
ax_gsea.set_title("D  GSEA — Top 15 Negatively Enriched Immune Pathways",
                  fontweight='bold', fontsize=11, loc='left')
ax_gsea.spines[['top','right']].set_visible(False)

# ─── Panel E  IRS distribution ───────────────────────────────────────────────
bins = np.linspace(cohort['IRS'].min(), cohort['IRS'].max(), 35)
ax_irs.hist(cohort.loc[high_mask, 'IRS'], bins=bins,
            color=HIGH_COL, alpha=0.65, label='High IRS', density=True)
ax_irs.hist(cohort.loc[low_mask,  'IRS'], bins=bins,
            color=LOW_COL,  alpha=0.65, label='Low IRS',  density=True)
ax_irs.axvline(median_irs, color='black', linestyle='--', linewidth=1.2,
               label=f'Median = {median_irs:.2f}')

# KDE overlay
from scipy.stats import gaussian_kde
x_range = np.linspace(cohort['IRS'].min(), cohort['IRS'].max(), 200)
for mask, col in [(high_mask, HIGH_COL), (low_mask, LOW_COL)]:
    kde = gaussian_kde(cohort.loc[mask, 'IRS'])
    ax_irs.plot(x_range, kde(x_range), color=col, linewidth=2)

ax_irs.set_xlabel("Immune Risk Score (standardised)", fontsize=10)
ax_irs.set_ylabel("Density", fontsize=10)
ax_irs.set_title("E  IRS Distribution", fontweight='bold', fontsize=11, loc='left')
ax_irs.legend(fontsize=8.5, framealpha=0.9)
ax_irs.spines[['top','right']].set_visible(False)

# ─── Panel F  Cox forest plot ────────────────────────────────────────────────
params   = cph.params_
ci_lower = cph.confidence_intervals_.iloc[:, 0]
ci_upper = cph.confidence_intervals_.iloc[:, 1]
pvals    = cox_summary['p']
hr_vals  = np.exp(params)
hr_lo    = np.exp(ci_lower)
hr_hi    = np.exp(ci_upper)

cov_labels = {
    'IRS'   : 'IRS (per SD)',
    'age'   : 'Age (per year)',
    'stage' : 'Tumour Stage'
}

y_cox = range(len(params))
cols_cox = [LOW_COL if hr < 1 else HIGH_COL for hr in hr_vals.values]

ax_cox.scatter(hr_vals.values, list(y_cox),
               s=100, color=cols_cox, zorder=5, edgecolors='white', linewidths=1)
for i, (lo, hi) in enumerate(zip(hr_lo.values, hr_hi.values)):
    ax_cox.hlines(i, lo, hi, colors=cols_cox[i], linewidth=2.5, alpha=0.85)

ax_cox.axvline(1.0, color='black', linestyle='--', linewidth=1.2, alpha=0.8)

ax_cox.set_yticks(list(y_cox))
ax_cox.set_yticklabels([cov_labels.get(n, n) for n in params.index], fontsize=10)
ax_cox.set_xlabel("Hazard Ratio (HR)  [95 % CI]", fontsize=10)
ax_cox.set_title("F  Cox PH Regression — Forest Plot",
                  fontweight='bold', fontsize=11, loc='left')

# Annotate HR + p
for i, (hr, lo, hi, pv) in enumerate(
        zip(hr_vals.values, hr_lo.values, hr_hi.values, pvals.values)):
    pstr_f = f"p={pv:.2e}" if pv >= 0.001 else f"p<0.001"
    ax_cox.text(max(hr_hi.values) * 1.08, i,
                f"HR={hr:.2f} ({lo:.2f}–{hi:.2f})\n{pstr_f}",
                va='center', fontsize=8.5, color='#333')

ax_cox.text(0.02, -0.12,
            f"C-index = {cph.concordance_index_:.3f}   |   n = {N_PATIENTS}",
            transform=ax_cox.transAxes, fontsize=9, color='#555')
ax_cox.spines[['top','right']].set_visible(False)

# ─────────────────────────────────────────────────────────────────────────────
# 12.  Save
# ─────────────────────────────────────────────────────────────────────────────
outfile = os.path.join(OUTDIR, "survival_analysis_plot.png")
fig.savefig(outfile, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close(fig)
print(f"\n[Figure saved] → {outfile}")

# ─────────────────────────────────────────────────────────────────────────────
# 13.  Statistical summary report
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  STATISTICAL JUSTIFICATION SUMMARY")
print("=" * 70)

print(f"""
Dataset overview
----------------
• DEA (full)       : {dea_full.shape[0]:,} tested genes; {len(sig_full):,} significant
                     (padj < 0.05, |log2FC| > 1)
  – Up-regulated   : {len(up_full):,}
  – Down-regulated : {len(down_full):,}
• DEA (3-sample)   : {dea_3.shape[0]:,} genes; {len(sig_3):,} significant
• GSEA             : {gsea.shape[0]:,} terms; {len(sig_gsea):,} FDR-significant
  – Negatively enriched immune pathways: {len(neg_gsea):,}
    Mean NES = {mean_neg_nes:.3f}

Kaplan–Meier analysis
---------------------
• Estimator        : Kaplan–Meier (Greenwood variance)
• Stratification   : High vs Low Immune Risk Score (median split)
• Log-rank test    : χ²({lr.test_statistic:.2f}), p = {lr.p_value:.2e}
  {'✓ SIGNIFICANT at α = 0.05' if lr.p_value < ALPHA else '✗ Not significant'}
• Median OS
  – High IRS : {med_high:.1f} months
  – Low  IRS : {med_low:.1f} months

Cox proportional-hazards model
--------------------------------
• Variables        : IRS (continuous), Age, Tumour Stage
• IRS hazard ratio : {irs_hr:.3f}  (95 % CI: {irs_ci[0]:.3f}–{irs_ci[1]:.3f})
  p = {irs_pval:.2e}  {'✓ SIGNIFICANT' if irs_pval < ALPHA else '✗ NS'}
• Concordance index: {cph.concordance_index_:.4f}
• PH assumption    : tested by Schoenfeld residuals (see above)

Interpretation
--------------
Higher IRS (reflecting intact immune activity) is associated with
HR < 1, i.e. lower hazard / better survival — consistent with the
GSEA finding that immune pathways are negatively enriched (NES < 0)
in the comparison group, indicating immune suppression and expected
worse clinical outcome.
""")

print("[All outputs written to:", OUTDIR, "]")
