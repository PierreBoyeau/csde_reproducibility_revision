import json

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

from .config import MERFISH_PANCANCER_DIR


def compute_dist_to_nn(adata, group_key, group_value):
    """Distance from every cell to its nearest neighbor within a given cell group."""
    pop_ref = adata.obs[group_key].str.contains(group_value).values.astype(bool)
    ref_adata = adata[pop_ref].copy()
    ref_ = NearestNeighbors(n_neighbors=1)
    ref_.fit(ref_adata.obsm["pos"])
    d_to_nn = ref_.kneighbors(adata.obsm["pos"], return_distance=True)[0].flatten()
    return ref_, d_to_nn


def add_spatial_group(
    adata, spatial_neighbor_name, spatial_dist_threshold=20, group_key="cell_type"
):
    """Add a binary spatial_group column (1 = inside, 0 = outside) based on distance to a cell type."""
    adata.obsm["pos"] = adata.obs[["centroid_x", "centroid_y"]].values
    _, d_to_nn = compute_dist_to_nn(
        adata, group_key=group_key, group_value=spatial_neighbor_name
    )
    adata.obs["d_to_nn"] = d_to_nn
    adata.obs["spatial_group"] = (d_to_nn < spatial_dist_threshold).astype(int)
    return adata


def load_annotations(path_to_annotations):
    """Load manual cell annotations from a CVAT-exported JSON file."""
    with open(path_to_annotations, "r") as f:
        annotations_dict = json.load(f)

    annotation_classes = {}
    for item_id, item in enumerate(annotations_dict["categories"]["label"]["labels"]):
        annotation_classes[item_id] = item["name"]

    ids, annots = [], []
    for item in annotations_dict["items"]:
        try:
            ids.append(item["id"])
            annots.append(item["annotations"][0]["label_id"])
        except Exception:
            print(item)

    return (
        pd.DataFrame({"id": np.array(ids), "annot": np.array(annots)})
        .assign(annotation_name=lambda x: x["annot"].map(annotation_classes))
        .drop_duplicates(subset=["id"])
        .set_index("id")
    )


def load_merfish_adata(dataset_name, spatial_neighbor_name, spatial_dist_threshold=20):
    """Load a MERFISH pancancer adata and attach binary spatial group labels."""
    path_to_adata = f"{MERFISH_PANCANCER_DIR}/{dataset_name}_adata.annotated.h5ad"
    adata = sc.read_h5ad(path_to_adata)
    return add_spatial_group(adata, spatial_neighbor_name, spatial_dist_threshold)


def prepare_cd8_t_cell_data(
    adata_path: str,
    annotations_path: str,
    n_cells_expressed_threshold: int = 10,
    annot_df: pd.DataFrame | None = None,
):
    """
    Prepare data for the CD8 T cell colocalization experiment.

    Returns
    -------
    dict
        ``adata_gt`` : AnnData
            Annotated cells with manual target/reference labels; filtered genes.
        ``adata_other`` : AnnData
            Unannotated cells, same ``.var`` as ``adata_gt``.
        ``adata_right_cells_imputed`` : AnnData
            Full-slide cells matching pred target/reference, same genes.
        ``is_gene_expressed`` : ndarray of bool
            Gene mask on the original object's ``.var`` axis.
        ``n_cells_expressing`` : ndarray of int
            Per-gene count of cells with count ≥ 1 among annotated cells in the
            pred union: CD8+ T cells inside the tumor, or any T cell outside.
    """
    spatial_group_key = "spatial_group_str"
    spatial_dist_threshold = 20

    # 1. Load data and compute spatial groups
    adata = sc.read_h5ad(adata_path)
    adata.obsm["pos"] = adata.obs[["centroid_x", "centroid_y"]].values

    _, d_to_nn = compute_dist_to_nn(adata, group_key="cell_type", group_value="cancer")
    adata.obs["d_to_nn"] = d_to_nn
    # 1 = Inside tumor boundary, 0 = Outside tumor boundary
    adata.obs[spatial_group_key] = (d_to_nn < spatial_dist_threshold).astype(int)

    # 2. Load manual annotations
    if annot_df is None:
        annot_df = load_annotations(annotations_path)
    annotated_cells = annot_df.index.intersection(adata.obs_names)

    adata.obs["hand_annot"] = np.nan
    adata.obs.loc[annotated_cells, "hand_annot"] = annot_df.loc[
        annotated_cells, "annotation_name"
    ].values

    # 3. Define CD8 subsets
    adata.obs.loc[:, "CD8A"] = adata[:, "CD8A"].layers["counts"].toarray().flatten()

    # Focus exclusively on CD8+ cells
    is_relevant_t_cell = lambda x: (x["CD8A"] >= 0.5)

    # 4. Create Prediction (Automated) Groups
    # 1 = Target (CD8+ T cell inside tumor)
    # 0 = Reference (any T cell outside tumor; matches legacy colocalization script)
    # 2 = Other
    adata.obs["prediction"] = 2

    is_target_pred = (
        is_relevant_t_cell(adata.obs)
        & (adata.obs[spatial_group_key] == 1)
        & (adata.obs["cell_type"] == "t cell")
    )
    ## MODE: ALL
    # is_reference_pred = (adata.obs[spatial_group_key] == 0) & (
    #     adata.obs["cell_type"] == "t cell"
    # )
    ## MODE: SAME
    is_reference_pred = (
        is_relevant_t_cell(adata.obs)
        & (adata.obs[spatial_group_key] == 0)
        & (adata.obs["cell_type"] == "t cell")
    )

    adata.obs.loc[is_target_pred, "prediction"] = 1
    adata.obs.loc[is_reference_pred, "prediction"] = 0

    # 5. Create Annotation (Manual) Groups for annotated cells
    adata_gt = adata[annotated_cells].copy()
    adata_gt.obs["annotation"] = 2

    is_target_manual = (
        is_relevant_t_cell(adata_gt.obs)
        & (adata_gt.obs[spatial_group_key] == 1)
        & (adata_gt.obs["hand_annot"] == "t cell")
    )
    ## MODE: ALL
    # is_reference_manual = (adata_gt.obs[spatial_group_key] == 0) & (
    #     adata_gt.obs["hand_annot"] == "t cell"
    # )
    ## MODE: SAME
    is_reference_manual = (
        is_relevant_t_cell(adata_gt.obs)
        & (adata_gt.obs[spatial_group_key] == 0)
        & (adata_gt.obs["hand_annot"] == "t cell")
    )

    adata_gt.obs.loc[is_target_manual, "annotation"] = 1
    adata_gt.obs.loc[is_reference_manual, "annotation"] = 0

    # 6. Filter for minimally expressed genes (pred target/ref on adata_gt, as
    # in the old adata_right_cells block); adata_other uses this same mask.
    is_target_pred_gt = (
        is_relevant_t_cell(adata_gt.obs)
        & (adata_gt.obs[spatial_group_key] == 1)
        & (adata_gt.obs["cell_type"] == "t cell")
    )
    is_reference_pred_gt = (adata_gt.obs[spatial_group_key] == 0) & (
        adata_gt.obs["cell_type"] == "t cell"
    )

    # todo: check
    # adata_right_cells = adata_gt[is_target_pred_gt | is_reference_pred_gt]
    adata_right_cells = adata_gt[is_target_manual | is_reference_manual]
    x_right = adata_right_cells.layers["counts"].astype(int).astype(float)
    n_cells_expressing = np.array((x_right >= 1).sum(0)).flatten()
    is_gene_expressed = n_cells_expressing >= n_cells_expressed_threshold

    adata_gt = adata_gt[:, is_gene_expressed].copy()

    # 7. Full-slide pred target/reference — independent gene filter based on these cells
    pair_mask = (is_target_pred | is_reference_pred).values
    _adata_imputed_all = adata[pair_mask]
    x_imputed = _adata_imputed_all.layers["counts"].astype(int).astype(float)
    n_cells_expressing_imputed = np.array((x_imputed >= 1).sum(0)).flatten()
    is_gene_expressed_imputed = n_cells_expressing_imputed >= n_cells_expressed_threshold
    adata_right_cells_imputed = _adata_imputed_all[:, is_gene_expressed_imputed].copy()
    adata_right_cells_old = _adata_imputed_all[:, is_gene_expressed].copy()

    # 8. Unannotated partition, same genes as adata_gt
    adata_other = adata[~adata.obs_names.isin(annotated_cells)][
        :, is_gene_expressed
    ].copy()

    return {
        "adata_gt": adata_gt,
        "adata_other": adata_other,
        "adata_right_cells_imputed": adata_right_cells_imputed,
        "adata_right_cells_gt": adata_right_cells,
        "is_gene_expressed": is_gene_expressed,
        "is_gene_expressed_imputed": is_gene_expressed_imputed,
        "n_cells_expressing": pd.Series(n_cells_expressing, index=adata.var_names),
        "n_cells_expressing_imputed": pd.Series(n_cells_expressing_imputed, index=adata.var_names),
        "adata_right_cells_old": adata_right_cells_old,
    }
