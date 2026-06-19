# %%
import os
from csde.model_poisson import PoissonIntercept

import numpy as np
import pandas as pd

from src.config import MERFISH_PANCANCER_DIR, MAIN_ADATA, MAIN_SAMPLE_NAME, RESULTS_DIR
from src.csde_analysis import build_model_inputs, run_one_model, run_imputation_baseline
from src.merfish_pancancer_data import prepare_cd8_t_cell_data

n_cells_expressed_threshold = 10
adata_path = os.path.join(MERFISH_PANCANCER_DIR, MAIN_ADATA)
annotations_path = os.path.join(
    MERFISH_PANCANCER_DIR,
    MAIN_SAMPLE_NAME,
    "manual_annotations",
    "annotations.json",
)
results_dir = os.path.join(RESULTS_DIR, "bias_vs_power")
os.makedirs(results_dir, exist_ok=True)

# %%
_prep = prepare_cd8_t_cell_data(
    adata_path=adata_path,
    annotations_path=annotations_path,
    n_cells_expressed_threshold=n_cells_expressed_threshold,
)
adata_gt = _prep["adata_gt"]
adata_other = _prep["adata_other"]
adata_right_cells_imputed = _prep["adata_right_cells_imputed"]

# import scanpy as sc
# sc.pp.subsample(adata_other, n_obs=50000)

shared, gene_names = build_model_inputs(adata_gt, adata_other)

# %%
res_poisson = run_one_model(
    PoissonIntercept,
    lambd_=None,
    model_name="CSDE",
    shared=shared,
    gene_names=gene_names,
    class_kwargs=dict(lambd_mode="element"),
    optimizer_kwargs=dict(tol=1e-5, n_iter=2000),
)
res_poisson.to_csv(os.path.join(results_dir, "results_poisson.csv"), index=False)

# %%
adata_imputed = adata_right_cells_imputed.copy()
print(adata_imputed.obs["prediction"].value_counts())
adata_imputed.X = adata_imputed.layers["counts"].astype(int).astype(float)
res_imputation = run_imputation_baseline(adata_imputed, "prediction", family="poisson").assign(n_obs=adata_imputed.n_obs)
res_imputation.to_csv(os.path.join(results_dir, "results_imputation.csv"), index=False)

# %%
results = pd.concat([res_poisson, res_imputation])
results.to_csv(os.path.join(results_dir, "results.csv"), index=False)
