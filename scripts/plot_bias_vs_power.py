# %%
import os
import pandas as pd
from scipy import stats
import plotnine as gg
import numpy as np
from src.config import RESULTS_DIR, DEFAULT_THEME
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"

path_to_results = os.path.join(RESULTS_DIR, "bias_vs_power")

# %%
# results_path = os.path.join(path_to_results, "results.csv")
# results1 = pd.read_csv(results_path)
results1 = pd.read_csv(os.path.join(path_to_results, "results_poisson.csv"))
results2 = pd.read_csv(os.path.join(path_to_results, "results_imputation.csv"))

# %%
results1["gene_rank"] = results1["pval"].rank(method="first", ascending=True)
results2["gene_rank"] = results2["pval"].rank(method="first", ascending=True)
results1["abs_beta"] = np.abs(results1["beta"])
results2["abs_beta"] = np.abs(results2["beta"])
results1["gene_rank_by_abs_beta"] = results1["abs_beta"].rank(method="first", ascending=False)
results2["gene_rank_by_abs_beta"] = results2["abs_beta"].rank(method="first", ascending=False)

results1["std_norm"] = np.sqrt(results1["cov"])
results2["std_norm"] = np.sqrt(results2["cov"]) * np.sqrt(35677.0 / 600)

results = pd.concat([results1, results2])
results["is_significant"] = results["padj"] < 0.05


# %%
pivot_cols = [col for col in results.columns if col != "model" and col != "gene_name"]
if "is_significant_005" in results.columns:
    results["is_significant_005"] = results["is_significant_005"].astype("boolean")

results_pivot = results.pivot_table(
    index="gene_name",
    columns="model",
    values=pivot_cols,
    aggfunc="first",
    dropna=True,
)
results_pivot.columns = [f"{col}_{model}" for col, model in results_pivot.columns]
results_pivot = results_pivot.reset_index()
# results_pivot["beta_CSDE"] = results_pivot["beta_CSDE"].fillna(0.0)
# results_pivot["pval_CSDE"] = results_pivot["pval_CSDE"].fillna(1.0)
results_pivot.info()

# %%
results_pivot_inter = results_pivot.dropna(subset=["beta_CSDE", "beta_imputation"]).reset_index(drop=True)
results_pivot_inter["gene_rank_imputation"] = results_pivot_inter["beta_imputation"].abs().rank(method="first", ascending=False)
results_pivot_inter["gene_rank_csde"] = results_pivot_inter["beta_CSDE"].abs().rank(method="first", ascending=False)
results_pivot_inter["delta_se"] = (
    results_pivot_inter["std_norm_CSDE"] - results_pivot_inter["std_norm_imputation"]
)
results_pivot_inter["log_std_ratio"] = np.log(
    results_pivot_inter["std_norm_imputation"] / results_pivot_inter["std_norm_CSDE"]
)
results_pivot_inter["log_std_ratio"] = np.clip(results_pivot_inter["log_std_ratio"], np.quantile(results_pivot_inter["log_std_ratio"], 0.05), np.quantile(results_pivot_inter["log_std_ratio"], 0.95))
# results_pivot_inter["log_std_ratio"].isna().sum()

results_pivot_inter["log_abs_beta_ratio"] = np.log(
    results_pivot_inter["abs_beta_CSDE"] / results_pivot_inter["abs_beta_imputation"]
)
results_pivot_inter = results_pivot_inter.loc[lambda x: x["beta_CSDE"] > -5.]  #remove single outlier

# %%
results_pivot_inter_diff = results_pivot_inter.loc[
    lambda x: x["is_significant_imputation"] & ~x["is_significant_CSDE"]
]
results_pivot_inter_diff

# %%
results_pivot_inter_diff[["beta_CSDE", "beta_imputation", "std_norm_CSDE", "std_norm_imputation", "log_std_ratio", "log_abs_beta_ratio"]].sample(30)


# %%
import numpy as np
import pandas as pd

x_range = np.linspace(results_pivot_inter_diff["beta_CSDE"].min(),
                    results_pivot_inter_diff["beta_CSDE"].max(), 200)
ribbon_df = pd.DataFrame({"x": x_range, "ymin": x_range - np.log(2), "ymax": x_range + np.log(2)})

fig = (
    gg.ggplot(results_pivot_inter_diff, gg.aes(x="beta_CSDE", y="beta_imputation", color="log_std_ratio"))
    + gg.geom_ribbon(gg.aes(x="x", ymin="ymin", ymax="ymax"), data=ribbon_df,
                    fill="lightgrey", alpha=0.5, inherit_aes=False)
    + gg.geom_point(size=1.0, stroke=0)
    + gg.geom_abline(intercept=-np.log(2), slope=1, linetype="dashed", size=0.1)
    + gg.geom_abline(intercept=np.log(2), slope=1, linetype="dashed", size=0.1)
    + gg.geom_hline(yintercept=0, size=0.1)
    + gg.geom_vline(xintercept=0, size=0.1)
    + DEFAULT_THEME
    + gg.scale_x_continuous(expand=(0.0, 0.0))
    + gg.scale_y_continuous(expand=(0.0, 0.0))
    + gg.theme(figure_size=(2.8, 2))
    + gg.labs(x="LFC (CSDE)", y="LFC (imputation)", color="log-SE ratio")
)
fig.save(os.path.join(path_to_results, "beta_scatter_diff.svg"))
print(stats.pearsonr(results_pivot_inter_diff["beta_CSDE"], results_pivot_inter_diff["beta_imputation"]))
fig

# %%
(results_pivot_inter_diff["log_abs_beta_ratio"].abs() >= np.log(2)).sum()

# %%
results_pivot_inter_diff[results_pivot_inter_diff["log_abs_beta_ratio"].abs() >= np.log(2)].loc[lambda x: np.sign(x["beta_CSDE"]) != np.sign(x["beta_imputation"])]

# %%
# %%
fig = (
    gg.ggplot(results_pivot_inter_diff, gg.aes(x="log_abs_beta_ratio", y="log_std_ratio"))
    + gg.geom_point(size=1.0, stroke=0)
    + DEFAULT_THEME
    + gg.theme(
        figure_size=(4, 2),
    )
)
fig

# %%
fig = (
    gg.ggplot(results_pivot_inter, gg.aes(x="gene_rank_csde", y="gene_rank_imputation"))
    + gg.geom_point(size=1.0, stroke=0)
    + DEFAULT_THEME
    + gg.theme(
        figure_size=(2, 2),
    )
    + gg.labs(
        x="CSDE rank",
        y="Imputation rank"
    )
)
stats.spearmanr(results_pivot_inter["gene_rank_csde"], results_pivot_inter["gene_rank_imputation"])

# %%
results_pivot_inter["delta_abs_beta"] = (
    results_pivot_inter["abs_beta_imputation"] - results_pivot_inter["abs_beta_CSDE"]
)
results_pivot_inter["delta_se"] = (
    results_pivot_inter["std_norm_CSDE"] - results_pivot_inter["std_norm_imputation"]
)

# %%
(
    gg.ggplot(results_pivot_inter, gg.aes(x="beta_CSDE", y="beta_imputation"))
    + gg.geom_point(size=1.0, stroke=0)
    + gg.geom_abline()
    + DEFAULT_THEME
)

# %%
results_pivot_inter["delta_se"].describe()

# %%
(
    gg.ggplot(results_pivot_inter, gg.aes(x="delta_se", y="delta_abs_beta"))
    + gg.geom_point(size=1.0, stroke=0)
    + gg.scale_x_continuous(limits=(0, 1))
    + DEFAULT_THEME
)


# %%

# %%

filtered_by_csde = results_pivot["beta_CSDE"].isna()
sig_csde = results_pivot["padj_CSDE"] < 0.05
sig_imp  = results_pivot["padj_imputation"] < 0.05

results_pivot["discovery_status"] = pd.Categorical(
    np.select(
        [filtered_by_csde, sig_csde & sig_imp, ~sig_csde & sig_imp, sig_csde & ~sig_imp],
        ["filtered_by_CSDE", "both", "automated_only", "CSDE_only"],
        default="neither",
    ),
    categories=["neither", "both", "automated_only", "CSDE_only", "filtered_by_CSDE"],
    ordered=True,
)

results_pivot["delta_abs_beta"] = (
    results_pivot["abs_beta_imputation"] - results_pivot["abs_beta_CSDE"]
)
results_pivot["delta_se"] = (
    results_pivot["std_norm_CSDE"] - results_pivot["std_norm_imputation"]
)

print(results_pivot["discovery_status"].value_counts())

# %%

