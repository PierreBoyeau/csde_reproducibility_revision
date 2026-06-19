# %%
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
from scipy import sparse
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.stats.multitest import multipletests

import os

RESULTS_DIR = os.environ["BC_RESULTS_DIR"]

sdata = sd.read_zarr(os.path.join(os.environ["BC_DATA_DIR"], "region_R2_annotated.zarr"))
adata = sdata["table"]
# %%
adata_macrophages = adata[adata.obs["cell_type"] == "macrophages"].copy()
adata_macrophages.obs["is_in_tumor"] = adata_macrophages.obs["is_in_tumor"] == "True"
adata_macrophages.obs["is_in_tumor"].value_counts()

# %%
n_cells_expressing = (adata_macrophages.X > 0).sum(0)
genes_to_keep = n_cells_expressing >= 10
adata_macrophages = adata_macrophages[:, genes_to_keep].copy()

# %%
def poisson_glm_comparison(adata, group_key):
    """Poisson GLM DE: compare each gene between group 1 vs 0 in group_key."""
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    X = X.astype(float)

    group = adata.obs[group_key].values.astype(float)
    design = np.column_stack([np.ones(len(group)), group])  # intercept + condition

    n_genes = X.shape[1]
    gene_names = adata.var_names
    results = []

    for i in range(n_genes):
        y = X[:, i]
        try:
            model = Poisson(y, design)
            fit = model.fit(disp=False, method="bfgs", maxiter=200)
            lfc = fit.params[1] / np.log(2)
            pval = fit.pvalues[1]
        except Exception:
            lfc, pval = np.nan, np.nan
        results.append((gene_names[i], lfc, pval))

    df = pd.DataFrame(results, columns=["gene_name", "lfc", "pval"])
    valid = df["pval"].notna()
    padj = np.full(len(df), np.nan)
    if valid.sum() > 0:
        padj[valid] = multipletests(df.loc[valid, "pval"], method="fdr_bh")[1]
    df["padj"] = padj
    return df.set_index("gene_name")

de_results = poisson_glm_comparison(adata_macrophages, "is_in_tumor").assign(
    abs_lfc=lambda df: df["lfc"].abs()
)
de_results.to_csv(f"{RESULTS_DIR}/imputation_poisson_results.csv", index=True)

# %%
de_hits = de_results.query("padj < 0.05").sort_values("abs_lfc", ascending=False)
de_hits.head(20).loc[:, ["lfc", "padj"]]
# %%
# de_results.dropna().sort_values("padj").query("padj < 0.1")
# %%
(de_results["padj"] <= 0.05).sum()
# %%
de_results["abs_lfc"]
# %%
csde_discoveries = [
    "ANPEP", "EPHB3", "CD163",
    "LYZ",
    "FCGR3A", "CX3CR1", "C3", "YAP1",
    "CLEC5A", "CSF3R", "DNMT3A", "TREM2"
]
# %%
de_results.loc[csde_discoveries, ["lfc", "padj"]]
# %%
