# %%
import os
import pandas as pd
import numpy as np
from scipy import stats
import plotnine as gg
from src.config import RESULTS_DIR, DEFAULT_THEME


def compute_jaccard(l1, l2):
    return np.intersect1d(l1, l2).shape[0] / np.union1d(l1, l2).shape[0]

# %%
RESULTS_DIR_EXPERIMENT = os.path.join(RESULTS_DIR, "poisson_vs_nb")

# %%
plot_df = pd.read_csv(
    os.path.join(RESULTS_DIR_EXPERIMENT, "adata_all_mean_variance_relationship.csv")
)

# %%
fig = (
    gg.ggplot(plot_df, gg.aes(x="gene_mean", y="phi"))
    + gg.geom_point(size=1.0, stroke=0)
    + gg.scale_x_log10()
    + gg.geom_hline(yintercept=0, size=0.1)
    + DEFAULT_THEME
    + gg.theme(
        figure_size=(2, 1.5),
    )
    # + gg.scale_y_log10()
    + gg.labs(
        x="mean",
        y="overdispersion"
    )
)
fig.save(os.path.join(RESULTS_DIR_EXPERIMENT, "overdispersion.svg"))
print(stats.spearmanr(plot_df["gene_mean"], plot_df["phi"]))
fig

# %%
poisson_df = pd.read_csv(os.path.join(RESULTS_DIR_EXPERIMENT, "results_poisson.csv"))
nb_df = pd.read_csv(os.path.join(RESULTS_DIR_EXPERIMENT, "results_nb.csv"))
poisson_df["gene_rank"] = poisson_df["pval"].rank(method="first", ascending=True)
nb_df["gene_rank"] = nb_df["pval"].rank(method="first", ascending=True)


results = pd.concat([poisson_df, nb_df])
pivot_cols = [col for col in results.columns if col != "model" and col != "gene_name"]
if "is_significant_005" in results.columns:
    results["is_significant_005"] = results["is_significant_005"].astype("boolean")

results_pivot = results.pivot_table(
    index="gene_name",
    columns="model",
    values=pivot_cols,
    aggfunc="first",
    dropna=False,
)
results_pivot.columns = [f"{col}_{model}" for col, model in results_pivot.columns]
results_pivot = results_pivot.reset_index()
results_pivot

# %%
fig = (
    gg.ggplot(results_pivot, gg.aes(x="beta_poisson", y="beta_nb"))
    + gg.geom_point(size=1.0, stroke=0)
    + gg.geom_abline(size=0.05)
    + DEFAULT_THEME
    + gg.theme(figure_size=(2.0, 1.5))
    + gg.labs(
        x="LFC~(Poisson model)",
        y="LFC~(NB model)"
    )
)
display(fig)
fig.save(os.path.join(RESULTS_DIR_EXPERIMENT, "beta_scatter.svg"))
print(stats.spearmanr(results_pivot["beta_poisson"], results_pivot["beta_nb"]))
print(stats.pearsonr(results_pivot["beta_poisson"], results_pivot["beta_nb"]))
# %%
max_y = results_pivot["gene_rank_poisson"].max()
y_breaks = range(0, int(max_y) + 25, 25)
fig = (
    gg.ggplot(results_pivot, gg.aes(x="gene_rank_nb", y="gene_rank_poisson"))
    + gg.geom_point(size=1.0, stroke=0)
      + gg.scale_x_continuous(expand=(0,0), limits=(0, 100), breaks=range(0, 101, 25))
      + gg.scale_y_continuous(
          expand=(0,0), 
          breaks=y_breaks,
        #   minor_breaks=range(25, y_max + 25, 50),    # gridlines only (no labels)
        )
    + DEFAULT_THEME
      + gg.theme(
          figure_size=(2.0, 1.5),
          panel_grid_major=gg.element_line(color="grey", size=0.1),
          panel_grid_minor=gg.element_blank(),
      )
      + gg.labs(
          x="gene rank (NB)",
          y="gene rank (Poisson)"
      )
)
fig.save(os.path.join(RESULTS_DIR_EXPERIMENT, "ranks.svg"))

# %%
is_de_poisson = poisson_df.query("padj <= 0.2")["gene_name"].unique()
is_de_nb = nb_df.query("padj <= 0.2")["gene_name"].unique()
np.intersect1d(is_de_poisson, is_de_nb).shape, np.union1d(is_de_poisson, is_de_nb).shape# %%

# %%
records = []
for k in [10, 15, 20, 25, 30]:
    topk_poisson = poisson_df.sort_values(by="pval")[:k]["gene_name"].unique()
    topk_nb = nb_df.sort_values(by="pval")[:k]["gene_name"].unique()
    jaccard = compute_jaccard(topk_poisson, topk_nb)
    records.append({"k": k, "jaccard": jaccard})

jaccard_df = pd.DataFrame(records)
jaccard_df["k"] = jaccard_df["k"].astype(str)

y_breaks = np.arange(0.0, 1.0, 0.1)  # increment of 0.05
print(y_breaks)
fig = (
    gg.ggplot(jaccard_df, gg.aes(x="k", y="jaccard"))
    + gg.geom_col(width=0.6)
    + gg.scale_y_continuous(expand=(0, 0), limits=(0, 1), breaks=y_breaks)
    + gg.labs(x="top K genes", y="Jaccard similarity")
    + DEFAULT_THEME
    + gg.theme(
        figure_size=(2.0, 1.5),
        panel_grid_major_y=gg.element_line(color="grey", size=0.1),
    )
)
fig
# %%

# %%
poisson_genes = poisson_df.query("is_significant_005")["gene_name"]
nb_genes = nb_df.query("is_significant_005")["gene_name"]
# %%
print(poisson_genes.shape)
print(nb_genes.shape)
print(compute_jaccard(poisson_genes, nb_genes))

