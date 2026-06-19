# %%
import os
import pandas as pd
from src.utils import get_results_dir, select_gpus

# select_gpus(n_gpus=1)

import pandas as pd
from csde.model_nb import NBIntercept
from csde.model_poisson import PoissonIntercept

from src.config import MERFISH_PANCANCER_DIR, MAIN_ADATA, MAIN_SAMPLE_NAME, RESULTS_DIR
from src.csde_analysis import build_model_inputs, run_one_model
from src.merfish_pancancer_data import prepare_cd8_t_cell_data

adata_path = os.path.join(MERFISH_PANCANCER_DIR, MAIN_ADATA)
annotations_path = os.path.join(
    MERFISH_PANCANCER_DIR,
    MAIN_SAMPLE_NAME,
    "manual_annotations",
    "annotations.json",
)
results_dir = os.path.join(RESULTS_DIR, "poisson_vs_nb")
os.makedirs(results_dir, exist_ok=True)

# %%
_prep = prepare_cd8_t_cell_data(
    adata_path=adata_path,
    annotations_path=annotations_path,
    n_cells_expressed_threshold=10,
)
adata_gt = _prep["adata_gt"]
adata_other = _prep["adata_other"]
# %%
X = adata_gt.layers["counts"].astype(int)
gene_mean = X.mean(axis=0)
gene_var = X.var(axis=0)
phi = (gene_var - gene_mean) / (gene_mean**2 + 1e-8)


plot_df = pd.DataFrame(
    {
        "gene_mean": gene_mean,
        "gene_var": gene_var,
        "phi": phi,
        "gene_name": adata_gt.var_names,
    }
)
plot_df.to_csv(
    os.path.join(results_dir, "adata_all_mean_variance_relationship.csv"), index=False
)
# %%
_prep = prepare_cd8_t_cell_data(
    adata_path=adata_path,
    annotations_path=annotations_path,
)
adata_gt = _prep["adata_gt"]
adata_other = _prep["adata_other"]



# %%
shared, gene_names = build_model_inputs(adata_gt, adata_other)

# %%
res_poisson = run_one_model(
    PoissonIntercept,
    lambd_=None,
    model_name="poisson",
    shared=shared,
    gene_names=gene_names,
    class_kwargs=dict(lambd_mode="element"),
    optimizer_kwargs=dict(tol=1e-5, n_iter=3000),
)
# %%
res_nb = run_one_model(
    NBIntercept,
    lambd_=None,
    model_name="nb",
    shared=shared,
    gene_names=gene_names,
    class_kwargs=dict(lambd_mode="element"),
    optimizer_kwargs=dict(tol=1e-5, n_iter=3000),
)
# res_nb.to_csv(os.path.join(results_dir, "results_nb.csv"), index=False)

# %%
results = pd.concat([res_poisson, res_nb])
results.to_csv(os.path.join(results_dir, "results.csv"), index=False)
