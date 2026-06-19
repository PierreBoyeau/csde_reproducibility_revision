import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

# from csde.model import InterceptRegression as CSDE

# DEFAULT_OPTIMIZER_KWARGS = dict(tol=1e-5, n_iter=2000, jit=True, optimizer="gd")
DEFAULT_OPTIMIZER_KWARGS = dict(tol=1e-3, n_iter=10000, jit=True, optimizer="gd")


def normalize_importance_weights(adata_gt):
    """Normalize inverse-sampling-probability weights to sum to n_cells."""
    w = 1.0 / adata_gt.obs["sampling_weight"].values
    return float(adata_gt.shape[0]) * w / w.sum()


def filter_expressed_genes(adata, min_cells=10, counts_layer="counts"):
    """Boolean mask of genes expressed in at least min_cells cells."""
    x = adata.layers[counts_layer]
    n_expressing = np.array((x >= 1).sum(0)).flatten()
    return n_expressing >= min_cells


def _glm_t_test(adata, idx, label_key, family="poisson"):
    if family == "poisson":
        family_ = sm.families.Poisson()
    elif family == "gaussian":
        family_ = sm.families.Gaussian()
    elif family == "nb":
        family_ = sm.families.NegativeBinomial()
    else:
        raise ValueError(f"Family {family} not supported")

    y = adata.X[:, idx]
    gene_name = adata.var_names[idx]
    x = adata.obs[label_key].astype(float).values.reshape(-1, 1)
    x_ = sm.add_constant(x)
    res = sm.GLM(y, x_, family=family_).fit()
    R = np.array([0, 1])
    res_sum = res.t_test(R)
    return {
        "pval": res_sum.pvalue.item(),
        "gene_name": gene_name,
        "beta": res.params[1],
        "cov": res.cov_params()[1, 1],
    }


def glm_test(adata, label_key, family="poisson"):
    res = []
    for idx in tqdm(range(adata.X.shape[1])):
        try:
            res_ = _glm_t_test(adata, idx, label_key, family)
        except Exception:
            res_ = {
                "pval": 1.0,
                "gene_name": adata.var_names[idx],
                "beta": 0.0,
                "cov": 0.0,
            }
        res.append(res_)
    res = pd.DataFrame(res)
    res["padj"] = multipletests(res["pval"], method="fdr_bh")[1]
    return res


def run_imputation_baseline(adata_imputed, spatial_group_key, family="poisson"):
    """Run GLM on imputed data and return results in CSDE-compatible format."""
    res = glm_test(adata_imputed, label_key=spatial_group_key, family=family)
    return res[["gene_name", "beta", "cov", "pval", "padj"]].assign(model="imputation")


def build_model_inputs(adata_gt, adata_other, optimizer_kwargs=None):
    """Extract arrays and shared constructor kwargs from AnnData objects."""
    if optimizer_kwargs is None:
        optimizer_kwargs = DEFAULT_OPTIMIZER_KWARGS.copy()
    y_gt = adata_gt.obs["annotation"].values.astype(int)
    y_hat = adata_gt.obs["prediction"].values.astype(int)
    y_unl = adata_other.obs["prediction"].values.astype(int)
    x_gt = adata_gt.layers["counts"].astype(int).astype(float)
    x_hat = adata_gt.layers["counts"].astype(int).astype(float)
    x_unl = adata_other.layers["counts"].astype(int).astype(float)
    shared = dict(
        inputs_gt=(x_gt, y_gt),
        inputs_hat=(x_hat, y_hat),
        inputs_unl=(x_unl, y_unl),
        importance_weights=normalize_importance_weights(adata_gt),
        optimizer_kwargs=optimizer_kwargs,
    )
    return shared, adata_gt.var_names


def run_one_model(
    model_class,
    lambd_,
    model_name,
    shared,
    gene_names,
    class_kwargs=None,
    optimizer_kwargs=None,
):
    """Fit one model configuration and return a tagged results DataFrame."""
    shared_local = dict(shared)
    if optimizer_kwargs is not None:
        shared_local["optimizer_kwargs"] = optimizer_kwargs
    elif shared_local.get("optimizer_kwargs") is None:
        shared_local["optimizer_kwargs"] = DEFAULT_OPTIMIZER_KWARGS.copy()
    model = model_class(**shared_local, **(class_kwargs or {}))
    model.fit(lambd_=lambd_)
    model.get_asymptotic_distribution()
    return model.test_differential_expression(1).assign(
        gene_name=gene_names, model=model_name
    )


# def run_csde_analysis(
#     adata_gt,
#     adata_other,
#     annotation_key="annotation",
#     prediction_key="prediction",
#     family="poisson",
#     optimizer_kwargs=None,
# ):
#     """Fit classic, imputation, and CSDE models; return concatenated results DataFrame."""
#     y_gt = adata_gt.obs[annotation_key].values.astype(int)
#     y_gt_pred = adata_gt.obs[prediction_key].values.astype(int)
#     y_pred = adata_other.obs[prediction_key].values.astype(int)

#     x_gt = adata_gt.layers["counts"].astype(int).astype(float)
#     x_gt_pred = adata_gt.layers["counts"].astype(int).astype(float)
#     x_pred = adata_other.layers["counts"].astype(int).astype(float)

#     gene_names = adata_gt.var_names
#     normalized_weights = normalize_importance_weights(adata_gt)
#     if optimizer_kwargs is None:
#         optimizer_kwargs = DEFAULT_OPTIMIZER_KWARGS.copy()

#     shared = dict(
#         inputs_gt=(x_gt, y_gt),
#         inputs_hat=(x_gt_pred, y_gt_pred),
#         inputs_unl=(x_pred, y_pred),
#         importance_weights=normalized_weights,
#         family=family,
#         optimizer_kwargs=optimizer_kwargs,
#     )

#     classic_model = CSDE(**shared)
#     classic_model.fit(lambd_=0.0)
#     classic_model.get_asymptotic_distribution()
#     res_classic = classic_model.test_differential_expression(1).assign(
#         gene_name=gene_names, model="classic"
#     )

#     imput_model = CSDE(**shared)
#     imput_model.fit(lambd_=1.0)
#     imput_model.get_asymptotic_distribution()
#     res_imput = imput_model.test_differential_expression(1).assign(
#         gene_name=gene_names, model="imputation"
#     )

#     csde_model = CSDE(**shared, lambd_mode="element")
#     csde_model.fit(lambd_=None)
#     csde_model.get_asymptotic_distribution()
#     res_csde = csde_model.test_differential_expression(1).assign(
#         gene_name=gene_names, model="csde"
#     )

#     return pd.concat([res_classic, res_imput, res_csde])
