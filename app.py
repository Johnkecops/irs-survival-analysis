#!/usr/bin/env python3
"""
Module: app
Purpose: Interactive Streamlit front-end for survival_analysis_pipeline.py (DEA + GSEA -> IRS -> KM/Cox) and PPI results
Author: Dr. Arli Aditya Parikesit
Date: 2026
Parameters (sidebar):
    - padj / |log2FC| cut-offs, GSEA FDR, cohort size, follow-up, seed, Cox penalizer
References:
    - Kaplan EL, Meier P (1958) J Am Stat Assoc 53:457-481
    - Cox DR (1972) J R Stat Soc B 34:187-220
    - Davidson-Pilon C (2019) lifelines. J Open Source Softw 4:1317
Run:
    streamlit run app.py
    python app.py   # self-check: defaults reproduce simulated_cohort.csv
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test

BASE = os.path.dirname(os.path.abspath(__file__))
HIGH_COL, LOW_COL = '#2166ac', '#d6604d'

# Same keyword list as survival_analysis_pipeline.py
IMMUNE_KEYWORDS = [
    'CD', 'HLA', 'IL', 'IFN', 'TNF', 'FCR', 'MHC', 'TLR', 'CXCL', 'CCL',
    'CXCR', 'CCR', 'IGH', 'IGL', 'IGK', 'TCR', 'TRAC', 'TRBC', 'PDCD',
    'CTLA', 'LAG', 'TIGIT', 'HAVCR', 'TNFRSF', 'TNFSF', 'NCR', 'NKG',
    'KLRD', 'KLRC', 'KLRB', 'BTK', 'SYK', 'LCK', 'ZAP70', 'PTPRC',
    'ITGAM', 'ITGAX', 'FOXP3', 'GZMB', 'PRF1', 'EOMES', 'TBX21'
]


def load_inputs(dea_src, gsea_src):
    dea = pd.read_csv(dea_src, index_col=0)
    gsea = pd.read_csv(gsea_src, index_col=0)
    missing = ({'log2FoldChange', 'padj', 'symbol', 'baseMean'} - set(dea.columns)) | \
              ({'Description', 'NES', 'p.adjust', 'setSize'} - set(gsea.columns))
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    dea = dea.rename(columns={'log2FoldChange': 'log2FC'})
    for col in ['NES', 'pvalue', 'p.adjust', 'setSize']:
        if col in gsea:
            gsea[col] = pd.to_numeric(gsea[col], errors='coerce')
    return dea, gsea


# Prefix match; CD/IL need a digit so CDC6, CDH*, ILK are excluded (same rule as pipeline)
IMMUNE_RE = re.compile('|'.join(k + r'\d' if k in ('CD', 'IL') else k for k in IMMUNE_KEYWORDS))


def is_immune(symbol):
    return pd.notna(symbol) and bool(IMMUNE_RE.match(str(symbol).upper()))


def simulate_cohort(dea, gsea, padj=0.05, lfc=1.0, gsea_fdr=0.05,
                    n=200, follow_up=60, seed=42):
    """Mirrors pipeline sections 2, 3, 5 (same RNG draw order -> identical cohort for defaults)."""
    np.random.seed(seed)
    sig = dea[(dea['padj'] < padj) & (dea['log2FC'].abs() > lfc)].copy()
    neg_gsea = gsea[(gsea['p.adjust'] < gsea_fdr) & (gsea['NES'] < 0)]
    if sig.empty or neg_gsea.empty:
        raise ValueError("No significant DEGs or negatively enriched GSEA terms at these cut-offs.")
    mean_neg_nes = neg_gsea['NES'].mean()

    immune = sig[sig['symbol'].apply(is_immune)]
    if immune.empty:
        immune = sig  # pipeline fallback: all significant DEGs
    weights = immune['log2FC'].values

    z = np.zeros((n, len(weights)))
    for g, w in enumerate(weights):
        z[:, g] = np.random.normal(-0.15 if w < 0 else 0.10, 1.0, n)
    irs = z @ weights
    irs = (irs - irs.mean()) / irs.std()
    group = np.where(irs >= np.median(irs), 'High', 'Low')

    beta = 0.55 * abs(mean_neg_nes) / 3.0
    # min-Gumbel errors -> Weibull AFT; intercept puts median at 36 mo
    t_true = np.exp(np.log(36) - 0.85 * np.log(np.log(2)) + beta * irs - 0.85 * np.random.gumbel(0, 1, n))
    cens = np.random.uniform(follow_up * 0.3, follow_up, n)
    ltfu = np.random.binomial(1, 0.15, n).astype(bool)
    t_obs = np.minimum(t_true, follow_up)
    event = (t_true <= follow_up).astype(int)
    t_obs[ltfu] = np.minimum(cens[ltfu], t_obs[ltfu])
    event[ltfu] = (t_true[ltfu] <= t_obs[ltfu]).astype(int)  # death before LTFU censor stays an event

    age = np.random.normal(60, 10, n).clip(20, 90)
    stage = np.random.choice([1, 2, 3, 4], p=[0.20, 0.30, 0.30, 0.20], size=n)
    cohort = pd.DataFrame({
        'patient_id': [f"PT{i:04d}" for i in range(n)], 'IRS': irs,
        'risk_group': group, 'time_months': t_obs, 'event': event,
        'age': age, 'stage': stage,
    })
    return cohort, sig, immune, neg_gsea


def fit_models(cohort, penalizer=0.01):
    hi, lo = cohort[cohort.risk_group == 'High'], cohort[cohort.risk_group == 'Low']
    kms = {name: KaplanMeierFitter().fit(d.time_months, d.event, label=name)
           for name, d in [('High IRS (immune-active)', hi), ('Low IRS (immune-suppressed)', lo)]}
    lr = logrank_test(hi.time_months, lo.time_months, hi.event, lo.event)
    cox_df = cohort[['time_months', 'event', 'IRS', 'age', 'stage']]
    cph = CoxPHFitter(penalizer=penalizer).fit(cox_df, 'time_months', 'event')
    ph = proportional_hazard_test(cph, cox_df, time_transform='rank')
    return kms, lr, cph, ph


# ─── Plots ───────────────────────────────────────────────────────────────────
def plot_km(kms, lr):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (name, km), col in zip(kms.items(), [HIGH_COL, LOW_COL]):
        km.plot_survival_function(ax=ax, ci_show=True, color=col, show_censors=True,
                                  censor_styles={'ms': 5, 'marker': '|'})
    ax.set(xlabel='Time (months)', ylabel='Survival probability', ylim=(0, 1.05))
    ax.text(0.02, 0.05, f"Log-rank p = {lr.p_value:.2e}", transform=ax.transAxes)
    ax.spines[['top', 'right']].set_visible(False)
    return fig


def plot_forest(cph):
    s = cph.summary
    hr, lo, hi = s['exp(coef)'], s['exp(coef) lower 95%'], s['exp(coef) upper 95%']
    fig, ax = plt.subplots(figsize=(7, 3))
    y = np.arange(len(s))
    cols = [LOW_COL if h < 1 else HIGH_COL for h in hr]
    ax.hlines(y, lo, hi, colors=cols, linewidth=2.5)
    ax.scatter(hr, y, color=cols, s=80, zorder=3)
    ax.axvline(1, color='k', ls='--', lw=1)
    ax.set_yticks(y, s.index)
    ax.set_xlabel('Hazard ratio (95% CI)')
    ax.spines[['top', 'right']].set_visible(False)
    return fig


def plot_volcano(dea, padj, lfc):
    v = dea.dropna(subset=['padj', 'log2FC']).copy()
    v['y'] = -np.log10(v['padj'].clip(lower=1e-300))
    sig = v['padj'] < padj
    fig, ax = plt.subplots(figsize=(7, 5))
    for mask, col, lab in [(~sig | (v.log2FC.abs() <= lfc), '#bbbbbb', 'NS'),
                           (sig & (v.log2FC > lfc), '#d6604d', 'Up'),
                           (sig & (v.log2FC < -lfc), '#2166ac', 'Down')]:
        ax.scatter(v.loc[mask, 'log2FC'], v.loc[mask, 'y'], s=4, c=col, alpha=0.6,
                   label=f"{lab} ({mask.sum():,})", rasterized=True)
    for _, r in v[sig & (v.log2FC.abs() > lfc)].nlargest(10, 'y').iterrows():
        ax.annotate(r['symbol'] if pd.notna(r['symbol']) else '', (r['log2FC'], r['y']), fontsize=7)
    ax.axhline(-np.log10(padj), ls='--', c='k', lw=0.8)
    for x in (-lfc, lfc):
        ax.axvline(x, ls='--', c='k', lw=0.8)
    ax.set(xlabel='log2 fold change', ylabel='-log10(padj)')
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    return fig


def plot_gsea(neg_gsea, k=15):
    d = neg_gsea.nsmallest(k, 'NES').sort_values('NES', ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(d) + 1))
    y = np.arange(len(d))
    ax.hlines(y, 0, d['NES'], color='#999')
    sc = ax.scatter(d['NES'], y, c=-np.log10(d['p.adjust']), cmap='Reds', s=70, zorder=3)
    ax.set_yticks(y, d['Description'].str[:55])
    ax.set_xlabel('NES')
    fig.colorbar(sc, ax=ax, label='-log10(FDR)')
    ax.spines[['top', 'right']].set_visible(False)
    return fig


# ─── Streamlit UI ────────────────────────────────────────────────────────────
def main():
    import streamlit as st

    st.set_page_config(page_title="IRS Survival Explorer", layout="wide")
    st.title("Immune Risk Score survival explorer")
    st.warning("Patient cohort is **simulated** from DEA/GSEA effect sizes. Survival "
               "estimates illustrate the method and are not clinical evidence; validate "
               "in a real annotated cohort (e.g. TCGA) before any prognostic claim.")

    with st.sidebar:
        st.header("Inputs")
        dea_up = st.file_uploader("DEA results (DESeq2 CSV with `symbol`)", type='csv')
        gsea_up = st.file_uploader("GSEA results (clusterProfiler CSV)", type='csv')
        st.header("Parameters")
        padj = st.number_input("DEG padj <", 1e-6, 0.5, 0.05, format="%.4f")
        lfc = st.slider("|log2FC| >", 0.0, 5.0, 1.0, 0.25)
        gsea_fdr = st.number_input("GSEA FDR <", 1e-6, 0.5, 0.05, format="%.4f")
        n = st.slider("Simulated patients", 50, 1000, 200, 50)
        follow_up = st.slider("Follow-up (months)", 12, 120, 60, 6)
        seed = st.number_input("Random seed", 0, 2**31 - 1, 42)
        pen = st.number_input("Cox penalizer", 0.0, 1.0, 0.01, format="%.3f")

    @st.cache_data
    def cached_load(dea_bytes, gsea_bytes):
        import io
        dea_src = io.BytesIO(dea_bytes) if dea_bytes else os.path.join(BASE, "DEA_Full_Results_macOS.csv")
        gsea_src = io.BytesIO(gsea_bytes) if gsea_bytes else os.path.join(BASE, "GSEA_Results_Detailed.csv")
        return load_inputs(dea_src, gsea_src)

    try:
        dea, gsea = cached_load(dea_up.getvalue() if dea_up else None,
                                gsea_up.getvalue() if gsea_up else None)
        cohort, sig, immune, neg_gsea = simulate_cohort(
            dea, gsea, padj, lfc, gsea_fdr, n, follow_up, int(seed))
        kms, lr, cph, ph = fit_models(cohort, pen)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()

    irs = cph.summary.loc['IRS']
    c = st.columns(5)
    c[0].metric("Significant DEGs", f"{len(sig):,}")
    c[1].metric("Immune DEGs (IRS)", f"{len(immune):,}")
    c[2].metric("Events", f"{cohort.event.sum()} / {len(cohort)}")
    c[3].metric("IRS HR (per SD)", f"{irs['exp(coef)']:.2f}",
                f"95% CI {irs['exp(coef) lower 95%']:.2f}-{irs['exp(coef) upper 95%']:.2f}",
                delta_color='off')
    c[4].metric("C-index", f"{cph.concordance_index_:.3f}")

    t_surv, t_dea, t_gsea, t_ppi, t_data = st.tabs(
        ["Survival", "DEA", "GSEA", "PPI network", "Cohort data"])

    with t_surv:
        a, b = st.columns(2)
        a.subheader("Kaplan-Meier")
        a.pyplot(plot_km(kms, lr))
        med = {k: km.median_survival_time_ for k, km in kms.items()}
        a.write(f"Log-rank chi2 = {lr.test_statistic:.2f}, p = {lr.p_value:.2e}  \n" +
                "  \n".join(f"Median OS, {k}: {v:.1f} mo" for k, v in med.items()))
        b.subheader("Cox PH (IRS + age + stage)")
        b.pyplot(plot_forest(cph))
        b.dataframe(cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']]
                    .style.format('{:.3g}'))
        b.caption("Schoenfeld residual test (PH assumption; p < 0.05 suggests violation)")
        b.dataframe(ph.summary.style.format('{:.3g}'))

    with t_dea:
        a, b = st.columns([3, 2])
        a.pyplot(plot_volcano(dea, padj, lfc))
        b.subheader("Immune DEGs used in IRS")
        b.dataframe(immune[['symbol', 'log2FC', 'padj', 'baseMean']]
                    .sort_values('log2FC'), height=450)

    with t_gsea:
        k = st.slider("Top terms", 5, 40, 15)
        st.pyplot(plot_gsea(neg_gsea, k))
        st.dataframe(neg_gsea[['Description', 'setSize', 'NES', 'p.adjust']].sort_values('NES'))

    with t_ppi:
        st.caption("Pre-computed by PPI-Analysis.R (STRINGdb + igraph); re-run the R script to update.")
        png = os.path.join(BASE, "PPI_Network_Plot.png")
        if os.path.exists(png):
            st.image(png)
        for f in ["PPI_hub_genes.csv", "PPI_centrality_metrics.csv"]:
            p = os.path.join(BASE, f)
            if os.path.exists(p):
                st.subheader(f)
                st.dataframe(pd.read_csv(p))

    with t_data:
        st.dataframe(cohort)
        st.download_button("Download cohort CSV", cohort.to_csv(index=False),
                           "simulated_cohort.csv", "text/csv")

    plt.close('all')  # st.pyplot keeps no reference; free figures each rerun


def _self_check():
    dea, gsea = load_inputs(os.path.join(BASE, "DEA_Full_Results_macOS.csv"),
                            os.path.join(BASE, "GSEA_Results_Detailed.csv"))
    cohort, *_ = simulate_cohort(dea, gsea)
    ref = pd.read_csv(os.path.join(BASE, "simulated_cohort.csv"))
    pd.testing.assert_frame_equal(cohort.reset_index(drop=True), ref, check_dtype=False)
    fit_models(cohort)
    print("OK: defaults reproduce simulated_cohort.csv; models fit")


if __name__ == "__main__":
    import sys
    if "streamlit" in sys.modules and hasattr(sys.modules["streamlit"], "runtime") \
            and sys.modules["streamlit"].runtime.exists():
        main()
    else:
        _self_check()
