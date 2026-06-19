# %%
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.environ["REPO_DIR"], "scripts"))

from csde import prepare_csde_inputs

from src.csde_analysis import build_model_inputs, run_one_model
from csde.model_poisson import PoissonIntercept

_bc = os.environ["BC_DATA_DIR"]
_bc_results = os.environ["BC_RESULTS_DIR"]
ANNOTATIONS_DIR = os.path.join(_bc, "annotations_macrophages")
# ANNOTATIONS_DIR = os.path.join(_bc, "annotations_macrophages_augmented")


# %%
_prep = prepare_csde_inputs(
    annotation_dir=ANNOTATIONS_DIR,
    spatial_group_target="in_tumor",
    spatial_group_reference="out_of_tumor",
    n_cells_expressed_threshold=10,
)

adata_gt = _prep["adata_gt"]
adata_other = _prep["adata_other"]

print(adata_gt)
print(adata_other)
# %%
shared, gene_names = build_model_inputs(adata_gt, adata_other)

# %%
res_current = run_one_model(
    PoissonIntercept,
    lambd_=None,
    model_name="current",
    shared=shared,
    gene_names=gene_names,
    class_kwargs=dict(lambd_mode="element"),
    optimizer_kwargs=dict(tol=1e-5, n_iter=2000),
)
# %%
res_current.to_csv(os.path.join(_bc_results, "csde_results.csv"), index=False)
de_genes = res_current.sort_values("padj").query("padj < 0.1")

upregulated_genes = de_genes.query("beta > 0")
print("Upregulated genes:")
print(", ".join(upregulated_genes["gene_name"]))

print("\nDownregulated genes:")
downregulated_genes = de_genes.query("beta < 0")
print(", ".join(downregulated_genes["gene_name"]))

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

# %%
res_classic.to_csv(os.path.join(_bc_results, "classic_poisson_results.csv"), index=False)
de_genes = res_classic.sort_values("padj").query("padj < 0.1")

upregulated_genes = de_genes.query("beta > 0")
print("Upregulated genes:")
print(", ".join(upregulated_genes["gene_name"]))

print("\nDownregulated genes:")
downregulated_genes = de_genes.query("beta < 0")
print(", ".join(downregulated_genes["gene_name"]))


# %%
