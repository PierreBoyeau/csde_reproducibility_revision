# %%
import numpy as np
import pandas as pd
import spatialdata as sd
import sys
from scipy import sparse
from statsmodels.discrete.discrete_model import Poisson
from statsmodels.stats.multitest import multipletests

import os
sys.path.insert(0, os.path.join(os.environ["REPO_DIR"], "scripts"))

from csde import prepare_csde_inputs
from src.csde_analysis import build_model_inputs, run_one_model
from csde.model_poisson import PoissonIntercept

_bc = os.environ["BC_DATA_DIR"]
ANNOTATIONS_DIR = os.path.join(_bc, "annotations_macrophages_replicate_v1")
SDATA_PATH = os.path.join(_bc, "region_R1_annotated.zarr")
RESULTS_DIR = os.environ["BC_RESULTS_DIR"]

# ── CSDE + classic ────────────────────────────────────────────────────────────
_prep = prepare_csde_inputs(
    annotation_dir=ANNOTATIONS_DIR,
    spatial_group_target="in_tumor",
    spatial_group_reference="out_of_tumor",
    n_cells_expressed_threshold=10,
)
adata_gt = _prep["adata_gt"]
adata_other = _prep["adata_other"]

adata_gt.layers["counts"] = adata_gt.X.copy()
adata_other.layers["counts"] = adata_other.X.copy()

shared, gene_names = build_model_inputs(adata_gt, adata_other)

# %%
res_csde = run_one_model(
    PoissonIntercept,
    lambd_=None,
    model_name="current",
    shared=shared,
    gene_names=gene_names,
    class_kwargs=dict(lambd_mode="element"),
    optimizer_kwargs=dict(tol=1e-5, n_iter=2000),
)
res_csde.to_csv(f"{RESULTS_DIR}/csde_results_replicate.csv", index=False)

# %%
shared_cp = shared.copy()
res_classic = run_one_model(
    PoissonIntercept,
    lambd_=0.0,
    model_name="current",
    shared=shared_cp,
    gene_names=gene_names,
    class_kwargs=dict(lambd_mode="overall"),
    optimizer_kwargs=dict(tol=1e-5, n_iter=2000),
)
res_classic.to_csv(f"{RESULTS_DIR}/classic_poisson_results_replicate.csv", index=False)

# ── Imputation (Poisson GLM on raw macrophage counts from replicate sdata) ────
def poisson_glm_comparison(adata, group_key):
    X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    X = X.astype(float)
    group = adata.obs[group_key].values.astype(float)
    design = np.column_stack([np.ones(len(group)), group])
    results = []
    for i in range(X.shape[1]):
        y = X[:, i]
        try:
            fit = Poisson(y, design).fit(disp=False, method="bfgs", maxiter=200)
            lfc, pval = fit.params[1] / np.log(2), fit.pvalues[1]
        except Exception:
            lfc, pval = np.nan, np.nan
        results.append((adata.var_names[i], lfc, pval))
    df = pd.DataFrame(results, columns=["gene_name", "lfc", "pval"])
    valid = df["pval"].notna()
    padj = np.full(len(df), np.nan)
    if valid.sum() > 0:
        padj[valid] = multipletests(df.loc[valid, "pval"], method="fdr_bh")[1]
    df["padj"] = padj
    return df.assign(abs_lfc=lambda d: d["lfc"].abs())

# %%
sdata = sd.read_zarr(SDATA_PATH)
adata_mac = sdata["table"]
adata_mac = adata_mac[adata_mac.obs["cell_type"] == "macrophages"].copy()
adata_mac.obs["is_in_tumor"] = (adata_mac.obs["is_in_tumor"].astype(str) == "True").astype(float)

# Restrict to CSDE replicate gene set, mirroring AUTOMATED_GENE_SUPPORT="csde" in the notebook.
support_genes = res_csde["gene_name"]
adata_mac_support = adata_mac[:, adata_mac.var_names.isin(support_genes)].copy()

# %%
res_imputation = poisson_glm_comparison(adata_mac_support, "is_in_tumor")
res_imputation.to_csv(f"{RESULTS_DIR}/imputation_poisson_results_replicate.csv", index=False)

print(f"CSDE replicate:       {len(res_csde)} genes, {(res_csde['padj'] < 0.1).sum()} sig (padj<0.1)")
print(f"Classic replicate:    {len(res_classic)} genes, {(res_classic['padj'] < 0.1).sum()} sig (padj<0.1)")
print(f"Imputation replicate: {len(res_imputation)} genes, {(res_imputation['padj'] < 0.1).sum()} sig (padj<0.1)")
