# %%
import json
import os
import anndata as ad
import pandas as pd
import spatialdata as sd

_merfish = os.environ["MERFISH_DIR"]
_bc = os.environ["BC_DATA_DIR"]

# %%
# --- helpers ---

def load_cvat_annotations(path):
    """Parse CVAT-style JSON -> pd.Series(cell_id -> label_name)."""
    with open(path) as f:
        ann = json.load(f)
    label_map = {
        i: cat["name"]
        for i, cat in enumerate(ann["categories"]["label"]["labels"])
    }
    records = {}
    for item in ann["items"]:
        if item["annotations"]:
            records[item["id"]] = label_map[item["annotations"][0]["label_id"]]
    return pd.Series(records, name="annotation")


def categorize_lung_colon(df, coi):
    """
    df must have columns 'cell_type' and 'annotation'.
    coi: cell type of interest (e.g. 't cell').
    The annotation space has exactly three labels: coi, one other specific type, 'other'.
    Returns a Series with values 'accept' / 'reject' / 'correct'.

    Rules:
      - accept : cell_type_3class == annotation  (agreement)
      - reject : cell_type_3class in {coi, specific_other} AND annotation == 'other'
      - correct: cell_type_3class != annotation  AND annotation != 'other'
                 (includes 'other' -> coi and 'other' -> specific_other)
    """
    ann_labels = set(df["annotation"].unique())
    specific_labels = ann_labels - {"other"}  # e.g. {'t cell', 'fibroblast'}

    def _3class(ct):
        return ct if ct in specific_labels else "other"

    ct3 = df["cell_type"].map(_3class)
    ann = df["annotation"]

    accept = ct3 == ann
    reject = (ct3 != ann) & (ann == "other")
    correct = (ct3 != ann) & (ann != "other")

    result = pd.Series("", index=df.index)
    result[accept] = "accept"
    result[reject] = "reject"
    result[correct] = "correct"
    return result


def compute_stats(df, coi, category_col="category"):
    """Print proportion validated and accept/correct/reject breakdown."""
    predicted_coi = df["cell_type"] == coi
    validated = predicted_coi & (df["annotation"] == coi)
    prop_validated = validated.sum() / predicted_coi.sum()
    print(f"  Predicted {coi!r}: {predicted_coi.sum()}")
    print(f"  Validated (annotation == coi): {validated.sum()}")
    print(f"  Proportion validated: {prop_validated:.3f}")
    print()
    if category_col in df.columns:
        cats = df[category_col].value_counts(normalize=True)
        print("  Category proportions:")
        for cat in ["accept", "correct", "reject"]:
            print(f"    {cat}: {cats.get(cat, 0):.3f}  (n={df[category_col].eq(cat).sum()})")


# %%
# ============================================================
# LUNG
# ============================================================
print("=" * 60)
print("LUNG")
print("=" * 60)

lung_adata = ad.read_h5ad(os.path.join(_merfish, "lung2_adata.annotated.h5ad"))
lung_ann = load_cvat_annotations(
    os.path.join(_merfish, "HumanLungCancerPatient2/manual_annotations/annotations.json")
)

lung_df = lung_adata.obs.loc[lung_ann.index, ["cell_type"]].copy()
lung_df["annotation"] = lung_ann
lung_df["category"] = categorize_lung_colon(lung_df, coi="t cell")

compute_stats(lung_df, coi="t cell")

# %%
# ============================================================
# COLON  (stored as liver2)
# ============================================================
print("=" * 60)
print("COLON")
print("=" * 60)

colon_adata = ad.read_h5ad(os.path.join(_merfish, "liver2_adata.annotated.h5ad"))
colon_ann = load_cvat_annotations(
    os.path.join(_merfish, "HumanLiverCancerPatient2/manual_annotations/annotations.json")
)

colon_df = colon_adata.obs.loc[colon_ann.index, ["cell_type"]].copy()
colon_df["annotation"] = colon_ann
colon_df["category"] = categorize_lung_colon(colon_df, coi="t cell")

compute_stats(colon_df, coi="t cell")

# %%
# ============================================================
# BREAST
# ============================================================
print("=" * 60)
print("BREAST")
print("=" * 60)

sdata = sd.read_zarr(os.path.join(_bc, "region_R2_annotated.zarr"))
breast_adata = sdata["table"]

with open(os.path.join(_bc, "annotations_macrophages/annotations.json")) as f:
    breast_ann_raw = json.load(f)

# True -> 'macrophages', False -> 'other'
breast_ann = pd.Series(
    {k: "macrophages" if v else "other" for k, v in breast_ann_raw.items()},
    name="annotation",
)

breast_df = breast_adata.obs.loc[breast_ann.index, ["cell_type"]].copy()
breast_df["annotation"] = breast_ann

# Only care about macrophage predictions: reject = predicted macrophage but annotated False.
# Anything with cell_type != macrophages is accept regardless of annotation.
is_mac_pred = breast_df["cell_type"] == "macrophages"
is_mac_ann = breast_df["annotation"] == "macrophages"
breast_df["category"] = "accept"
breast_df.loc[is_mac_pred & ~is_mac_ann, "category"] = "reject"

compute_stats(breast_df, coi="macrophages")

# %%
import matplotlib
import plotnine as gg

matplotlib.rcParams["svg.fonttype"] = "none"

def prop_validated(df, coi):
    predicted = df["cell_type"] == coi
    return (predicted & (df["annotation"] == coi)).sum() / predicted.sum()

plot_df = pd.DataFrame([
    {"dataset": "lung",   "pct_validated": 100 * prop_validated(lung_df,   "t cell")},
    {"dataset": "colon",  "pct_validated": 100 * prop_validated(colon_df,  "t cell")},
    {"dataset": "breast", "pct_validated": 100 * prop_validated(breast_df, "macrophages")},
])
plot_df["dataset"] = pd.Categorical(plot_df["dataset"], categories=["lung", "colon", "breast"])

print(plot_df)

fig = (
    gg.ggplot(plot_df, gg.aes(x="dataset", y="pct_validated"))
    + gg.geom_col(width=0.4, fill="#7EB8FF", color="none")
    + gg.scale_y_continuous(expand=(0, 0), limits=(0, 100))
    + gg.theme_bw()
    + gg.theme(
        figure_size=(1.8, 1.8),
        panel_grid_major_x=gg.element_blank(),
        panel_grid_minor=gg.element_blank(),
        axis_text=gg.element_text(size=8),
        axis_title=gg.element_text(size=9),
    )
    + gg.labs(x="", y="% cells validated")
)
fig.save("../results/annotation_validation_stats.svg")
fig
# %%
