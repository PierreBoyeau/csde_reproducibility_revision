import os

from src.utils import get_results_dir, select_gpus

# select_gpus(n_gpus=1)

import random

import numpy as np
import pandas as pd
import scanpy as sc

from csde.model_poisson import PoissonIntercept

from src.config import MERFISH_PANCANCER_DIR, MAIN_ADATA, MAIN_SAMPLE_NAME
from src.csde_analysis import build_model_inputs, run_one_model
from src.merfish_pancancer_data import prepare_cd8_t_cell_data

# Keep the script deterministic given the same seed.
RANDOM_SEEDS = [0, 1, 2, 3, 4]
SAMPLE_SIZES = [25, 50, 100, 200, 300, 400, 500]


adata_path = os.path.join(MERFISH_PANCANCER_DIR, MAIN_ADATA)
annotations_path = os.path.join(
    MERFISH_PANCANCER_DIR,
    MAIN_SAMPLE_NAME,
    "manual_annotations",
    "annotations.json",
)
results_dir = get_results_dir()


# %%
_prep = prepare_cd8_t_cell_data(
    adata_path=adata_path,
    annotations_path=annotations_path,
    n_cells_expressed_threshold=10,
)
adata_gt = _prep["adata_gt"]
adata_other = _prep["adata_other"]

for sample_size in SAMPLE_SIZES:
    for random_seed in RANDOM_SEEDS:
        random.seed(random_seed)
        np.random.seed(random_seed)
        sc.settings.seed = random_seed

        adata_gt_sub = adata_gt.copy()
        n_obs = int(min(sample_size, adata_gt_sub.n_obs))
        sc.pp.subsample(adata_gt_sub, n_obs=n_obs, random_state=random_seed)

        shared, gene_names = build_model_inputs(adata_gt_sub, adata_other)
        try:
            res_poisson = run_one_model(
                PoissonIntercept,
                lambd_=None,
                model_name="poisson",
                shared=shared,
                gene_names=gene_names,
                class_kwargs=dict(lambd_mode="element"),
                optimizer_kwargs=dict(tol=1e-5, n_iter=2000),
            ).assign(sample_size=n_obs, random_seed=random_seed)
            res_poisson.to_csv(
                os.path.join(
                    results_dir, f"poisson_sample_{n_obs}_seed_{random_seed}.csv"
                ),
                index=False,
            )
        except Exception as e:
            print(
                f"Error running Poisson model for sample size {n_obs} and random seed {random_seed}: {e}"
            )
            continue

# Baseline: full annotated set (no subsampling)
shared, gene_names = build_model_inputs(adata_gt, adata_other)
res_poisson_full = run_one_model(
    PoissonIntercept,
    lambd_=None,
    model_name="poisson",
    shared=shared,
    gene_names=gene_names,
    class_kwargs=dict(lambd_mode="element"),
    optimizer_kwargs=dict(tol=1e-5, n_iter=2000),
).assign(sample_size=int(adata_gt.n_obs), random_seed=0)
res_poisson_full.to_csv(
    os.path.join(results_dir, f"poisson_sample_{int(adata_gt.n_obs)}_seed_{0}.csv"),
    index=False,
)
