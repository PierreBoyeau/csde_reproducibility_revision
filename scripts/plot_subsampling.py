# %%
import os
import pandas as pd
import numpy as np
from scipy import stats
import plotnine as gg
from src.config import RESULTS_DIR, DEFAULT_THEME
import glob
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"

RESULTS_DIR_EXPERIMENT = os.path.join(RESULTS_DIR, "subsamplig")

# %%
results_paths = glob.glob(os.path.join(RESULTS_DIR_EXPERIMENT, "poisson_sample_*.csv"))
results = pd.concat([pd.read_csv(path) for path in results_paths])
results["is_significant"] = results["padj"] < 0.1
results["is_significant"] = results["is_significant"].fillna(False)
# %%
results_ref = results[results["sample_size"] == 600]
# %%
merged = results.merge(
    results_ref[["gene_name", "random_seed", "is_significant", "beta"]],
    on=["gene_name"],
    suffixes=("", "_ref"),
)


def compute_metrics(group):
    pred = group["is_significant"]
    true = group["is_significant_ref"]
    ndiscoveries = pred.sum()
    tp = (pred & true).sum()
    fp = (pred & ~true).sum()
    fn = (~pred & true).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    pearson_r, _ = stats.pearsonr(group["beta"], group["beta_ref"])
    return pd.Series({"precision": precision, "recall": recall, "pearson_r": pearson_r, "ndiscoveries": ndiscoveries})


metrics = (
    merged.groupby(["random_seed", "sample_size"])
    .apply(compute_metrics)
    .reset_index()
)
metrics["random_seed"].unique()

# %%
metrics_summary = (
    metrics.groupby("sample_size")[["precision", "recall", "pearson_r", "ndiscoveries"]]
    .agg(["mean", "std"])
    .reset_index()
)
metrics_summary.columns = ["_".join(c).strip("_") for c in metrics_summary.columns]
for col in ["precision", "recall", "pearson_r", "ndiscoveries"]:
    metrics_summary[f"{col}_ymin"] = metrics_summary[f"{col}_mean"] - metrics_summary[f"{col}_std"]
    metrics_summary[f"{col}_ymax"] = metrics_summary[f"{col}_mean"] + metrics_summary[f"{col}_std"]

def _mean_pairwise_jaccard(group):
    seeds = group["random_seed"].unique()
    print(seeds)
    if len(seeds) < 2:
        return np.nan
    sig_sets = {s: set(group.loc[group["random_seed"] == s, "gene_name"]) for s in seeds}
    jaccards = []
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i + 1:]:
            intersection = len(sig_sets[s1] & sig_sets[s2])
            union = len(sig_sets[s1] | sig_sets[s2])
            jaccards.append(intersection / union if union > 0 else 0.0)
    return np.mean(jaccards)

_sig = results[results["is_significant"]][["sample_size", "random_seed", "gene_name"]]
_jaccard = (
    _sig.groupby("sample_size")
    .apply(_mean_pairwise_jaccard)
    .reset_index(name="jaccard_mean")
)
metrics_summary = metrics_summary.merge(_jaccard, on="sample_size", how="left")

# %%
(
    gg.ggplot(metrics_summary, gg.aes(x="sample_size", y="precision_mean"))
    + gg.geom_point()
    + gg.geom_line()
    + gg.geom_errorbar(gg.aes(ymin="precision_ymin", ymax="precision_ymax"))
    + gg.labs(
        x="labeled sample size $n$",
        y="precision",
    )
    + DEFAULT_THEME
    + gg.theme(
        figure_size=(2.0, 1.5)
    )
)

# %%
fig = (
    gg.ggplot(metrics_summary, gg.aes(x="sample_size", y="pearson_r_mean"))
    + gg.geom_point()
    + gg.geom_line()
    + gg.geom_errorbar(gg.aes(ymin="pearson_r_ymin", ymax="pearson_r_ymax"))
    + gg.geom_hline(yintercept=0.8)
    + gg.labs(
        x="labeled sample size $n$",
        y="Pearson $r$",
    )
    + DEFAULT_THEME
    + gg.theme(
        figure_size=(2.0, 1.5)
    )
)
fig.save(os.path.join(RESULTS_DIR_EXPERIMENT, "pearsonr.svg"))
fig


# %%
fig = (
    gg.ggplot(metrics_summary, gg.aes(x="sample_size", y="recall_mean"))
    + gg.geom_point()
    + gg.geom_line()
    + gg.geom_errorbar(gg.aes(ymin="recall_ymin", ymax="recall_ymax"))
    + gg.labs(
        x="labeled sample size $n$",
        y="recall",
    )
    + DEFAULT_THEME
    + gg.theme(
        figure_size=(2.0, 1.5)
    )
)
fig.save

# %%
fig = (
    gg.ggplot(metrics_summary, gg.aes(x="sample_size", y="ndiscoveries_mean"))
    + gg.geom_point()
    + gg.geom_line()
    + gg.geom_hline(yintercept=metrics_summary.query("sample_size == 600")["ndiscoveries_mean"].item())
    + gg.geom_errorbar(gg.aes(ymin="ndiscoveries_ymin", ymax="ndiscoveries_ymax"))
    + gg.labs(
        x="labeled sample size $n$",
        y="# of detected DE genes",
    )
    + DEFAULT_THEME
    + gg.theme(
        figure_size=(2.0, 1.5)
    )
)
fig.save(os.path.join(RESULTS_DIR_EXPERIMENT, "n_degs.svg"))
fig

# %%
metrics_summary[["sample_size", "jaccard_mean"]]
# %%
