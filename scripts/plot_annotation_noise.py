# %%
import os
import pandas as pd
import numpy as np
import plotnine as gg
import glob
from src.config import RESULTS_DIR, DEFAULT_THEME
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"

RESULTS_DIR_EXPERIMENT = os.path.join(RESULTS_DIR, "annotation_noise")

# %%
files = glob.glob(os.path.join(RESULTS_DIR_EXPERIMENT, "*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df_ref = df.query("seed == 0 & noise_fraction == 0").copy()

# %%
order = ["0%", "1%", "3%", "5%", "10%"]


def noise_pct_label(noise_fraction_series):
    s = (noise_fraction_series * 100).astype(int).astype(str) + "%"
    return pd.Categorical(s, categories=order)


records = []
scatter_records = []
for (seed, noise_fraction), group in df.groupby(["seed", "noise_fraction"]):
    if noise_fraction > 0.1:
        continue
    merged = group.merge(df_ref, on="gene_name", suffixes=("_noise", "_ref"))

    sig_in_p1 = merged["padj_ref"] < 0.05
    sig_in_p2 = merged["padj_noise"] < 0.05

    sensitivity = (
        (-np.log10(merged.loc[sig_in_p1, "padj_noise"].clip(lower=1e-300))).median()
        if sig_in_p1.any() else np.nan
    )
    specificity = (
        (-np.log10(merged.loc[sig_in_p2, "padj_ref"].clip(lower=1e-300))).median()
        if sig_in_p2.any() else np.nan
    )

    sensitivity_pval = (
        merged.loc[sig_in_p1, "padj_noise"].median()
        if sig_in_p1.any() else np.nan
    )
    specificity_pval = (
        merged.loc[sig_in_p2, "padj_ref"].median()
        if sig_in_p2.any() else np.nan
    )

    records.append(
        {
            "seed": seed,
            "noise_fraction": noise_fraction,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "sensitivity_pval": sensitivity_pval,
            "specificity_pval": specificity_pval,
        }
    )

    tmp = merged[["gene_name", "padj_noise", "padj_ref"]].copy()
    tmp["seed"] = seed
    tmp["noise_fraction"] = noise_fraction
    scatter_records.append(tmp)

metrics_df = pd.DataFrame(records)
metrics_df["noise_pct"] = noise_pct_label(metrics_df["noise_fraction"])

scatter_df = pd.concat(scatter_records, ignore_index=True)
scatter_df["noise_pct"] = noise_pct_label(scatter_df["noise_fraction"])
scatter_df["log_p1"] = -np.log10(scatter_df["padj_ref"].clip(lower=1e-300))
scatter_df["log_p2"] = -np.log10(scatter_df["padj_noise"].clip(lower=1e-300))

# %%
metrics_long_pval = metrics_df.melt(
    id_vars=["noise_pct", "seed", "noise_fraction"],
    value_vars=["sensitivity_pval", "specificity_pval"],
    var_name="metric",
    value_name="value",
)
metrics_long_pval["metric"] = metrics_long_pval["metric"].str.replace("_pval", "")
metrics_long_pval = metrics_long_pval.loc[lambda x: x["noise_pct"].notna()]

fig = (
    gg.ggplot(metrics_long_pval, gg.aes(x="noise_pct", y="value"))
    + gg.geom_boxplot(width=0.5, outlier_size=0, size=0.4)
    + gg.geom_jitter(width=0.12, size=0.8, alpha=0.5)
    + gg.facet_wrap("~metric", nrow=1)
    + gg.geom_hline(yintercept=0.05, linetype="dashed", color="#AAAAAA", size=0.4)
    # + gg.scale_y_log10()
    + gg.labs(x="annotation noise", y="median adj-p")
    + DEFAULT_THEME
    + gg.theme(figure_size=(3.2, 1.4))
)
fig.save(os.path.join(RESULTS_DIR_EXPERIMENT, "sensitivity_specificity_pval_boxplot.svg"))
fig


# %%
