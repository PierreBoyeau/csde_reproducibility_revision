import os

import numpy as np
import pandas as pd
from csde.model_poisson import PoissonIntercept

from src.config import MERFISH_PANCANCER_DIR, MAIN_ADATA, MAIN_SAMPLE_NAME, RESULTS_DIR
from src.csde_analysis import build_model_inputs, run_one_model
from src.merfish_pancancer_data import load_annotations, prepare_cd8_t_cell_data

adata_path = os.path.join(MERFISH_PANCANCER_DIR, MAIN_ADATA)
annotations_path = os.path.join(
    MERFISH_PANCANCER_DIR,
    MAIN_SAMPLE_NAME,
    "manual_annotations",
    "annotations.json",
)
results_dir = os.path.join(RESULTS_DIR, "annotation_noise")
os.makedirs(results_dir, exist_ok=True)

base_annot_df = load_annotations(annotations_path)
annotation_names = base_annot_df["annotation_name"].unique().tolist()

NOISE_FRACTIONS = [0.0, 0.01,0.03,0.05, 0.1]
SEEDS = [0, 1, 2, 3, 4]

for seed in SEEDS:
    rng = np.random.default_rng(seed)
    for frac in NOISE_FRACTIONS:
        fname = f"results_seed{seed}_frac{int(frac * 100):03d}.csv"
        fpath = os.path.join(results_dir, fname)
        if os.path.exists(fpath):
            print(f"Skipping seed={seed}, frac={frac:.2f} -> {fname} exists")
            continue

        annot_df_noisy = base_annot_df.copy()
        names = annot_df_noisy["annotation_name"].values.copy()
        n_cells = len(names)
        n_flip = int(round(frac * n_cells))

        if n_flip > 0:
            flip_idx = rng.choice(n_cells, size=n_flip, replace=False)
            for i in flip_idx:
                current = names[i]
                other_classes = [c for c in annotation_names if c != current]
                names[i] = rng.choice(other_classes)

        annot_df_noisy["annotation_name"] = names

        _prep = prepare_cd8_t_cell_data(
            adata_path=adata_path,
            annotations_path=annotations_path,
            annot_df=annot_df_noisy,
        )
        adata_gt = _prep["adata_gt"]
        adata_other = _prep["adata_other"]
        shared, gene_names = build_model_inputs(adata_gt, adata_other)
        try:
            res = run_one_model(
                PoissonIntercept,
                lambd_=None,
                model_name="poisson",
                shared=shared,
                gene_names=gene_names,
                class_kwargs=dict(lambd_mode="element"),
                optimizer_kwargs=dict(tol=1e-5, n_iter=2000),
            )
            res = res.assign(seed=seed, noise_fraction=frac, n_flipped=n_flip)

            res.to_csv(fpath, index=False)
            print(f"Saved seed={seed}, frac={frac:.2f} -> {fname}")
        except Exception:
            print(f"Error for seed={seed}, frac={frac:.2f}")
