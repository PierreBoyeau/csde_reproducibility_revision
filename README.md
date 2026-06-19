# CSDE Revision Experiments

This repository contains the code for the experiments conducted during the revision of the CSDE paper.

## Configuration

Set these once in your shell before running any of the commands below:

```bash
export REPO_DIR="<path/to/csde_experiments_revision>"
export CSDE_DIR="<path/to/csde>"
export MERFISH_SRC="<path/to/merfish_pancancer_source>"
export MERFISH_DIR="<path/to/merfish_pancancer>"
export MERSCOPE_DIR="<path/to/merscope_data>"
export BC_DATA_DIR="<path/to/breast_cancer_data>"
export BC_RESULTS_DIR="<path/to/breast_cancer_results>"
```

- `REPO_DIR`: root of this git repo. Used to locate scripts and save outputs/figures.
- `CSDE_DIR`: a clone of the csde package repo. Only needed at setup time (`pip install -e`) and to run the interactive `export.py` / `annotate.py` tools in the breast cancer experiment.
- `MERSCOPE_DIR`: raw MERSCOPE output, containing one subdirectory per region (e.g. `region_R1/`, `region_R2/`), downloaded from the MERFISH 2.0 data portal ([download here](https://info.vizgen.com/merfish-2.0-data-release-form)). Read only by the preprocessing notebooks (`spatialdata_explore_R2.ipynb`, `spatialdata_explore_Rreplicate.ipynb`); not needed once the annotated zarr files have been written.
- `MERFISH_DIR`: directory containing the processed MERFISH pancancer dataset: `{sample}_adata.annotated.h5ad` files and `{SampleDir}/manual_annotations/annotations.json` subdirectories, obtained using the original paper reproducibility code [here](https://github.com/PierreBoyeau/csde_experiments). Used by all experiments in this repository except the breast cancer experiment.
- `BC_DATA_DIR`: staging directory for all breast cancer data. The preprocessing notebooks write `region_R1_annotated.zarr` and `region_R2_annotated.zarr` here; the Streamlit annotation tool writes `annotations_macrophages*/annotations.json` here. All three breast cancer experiment scripts (`breast_cancer_experiment.py`, `breast_cancer_experiment_all_replicates.py`, `breast_cancer_automated.py`) and `annotation_validation_stats.py` read from it.
- `BC_RESULTS_DIR`: output directory for breast cancer results CSVs (`csde_results.csv`, `classic_poisson_results.csv`, `imputation_poisson_results.csv`, and their `_replicate` variants). Written by the experiment scripts; does not need to pre-exist with content.

## Environment Setup

We recommend using a `conda` environment to manage dependencies. The setup below installs the local version of `csde` in editable mode, along with `spatialdata` and other required packages.

```bash
git clone https://github.com/YosefLab/csde.git $CSDE_DIR  # skip if already cloned
cd $CSDE_DIR && git checkout 31e1c09 && cd $REPO_DIR

conda create -n csde_revision python=3.12 pip -y
conda activate csde_revision
pip install -e "$CSDE_DIR[cuda12]"
conda install jupyterlab --yes
pip install -r extra_requirements.txt

python -c "import jax; print(jax.devices())"
```

## Relevant Scripts

```bash
conda activate csde_revision

# Poisson vs NB
python $REPO_DIR/scripts/poisson_vs_nb.py

# Subsampling
python $REPO_DIR/scripts/subsamplig.py

# Bias vs power
python $REPO_DIR/scripts/bias_vs_power.py

# Annotation noise
python $REPO_DIR/scripts/annotation_noise.py
```

## Data Preprocessing (spatialdata notebooks)

These notebooks load raw MERSCOPE data, cluster cells, annotate cell types, and write the annotated zarr files used downstream.

```bash
# scripts/spatialdata_explore_R2.ipynb       → produces region_R2_annotated.zarr
# scripts/spatialdata_explore_Rreplicate.ipynb → produces region_R1_annotated.zarr
```

## Breast Cancer Experiment

See also `scripts/breast_cancer/README.md` for more details.

### 1. Export initial annotation panels

```bash
conda activate csde_revision

SDATA_PATH="$BC_DATA_DIR/region_R2_annotated.zarr"
GENE_COLORS="$BC_DATA_DIR/gene_colors_file.json"
ANNOTATIONS_PATH="$BC_DATA_DIR/annotations_macrophages"

python $CSDE_DIR/scripts/export.py \
  --sdata $SDATA_PATH \
  --out $ANNOTATIONS_PATH \
  --cell-type-key cell_type \
  --cell-type-of-interest macrophages \
  --target-proportion 0.4 \
  --gene-colors $GENE_COLORS \
  --image-channel Cellbound2 \
  --n-cells 600 \
  --delta 15 \
  --n-top-genes 25
```

### 2. Annotate cells

```bash
ANNOTATIONS_PATH="$BC_DATA_DIR/annotations_macrophages"
streamlit run $CSDE_DIR/scripts/annotate.py -- --dir $ANNOTATIONS_PATH
```

### 3. Run CSDE experiment

```bash
python $REPO_DIR/scripts/breast_cancer/breast_cancer_experiment.py
```

### Replicate (R1)

Same steps with region R1 instead of R2:

```bash
SDATA_PATH="$BC_DATA_DIR/region_R1_annotated.zarr"
GENE_COLORS="$BC_DATA_DIR/gene_colors_file.json"
ANNOTATIONS_PATH="$BC_DATA_DIR/annotations_macrophages_replicate"

python $CSDE_DIR/scripts/export.py \
  --sdata $SDATA_PATH \
  --out $ANNOTATIONS_PATH \
  --cell-type-key cell_type \
  --cell-type-of-interest macrophages \
  --target-proportion 0.4 \
  --gene-colors $GENE_COLORS \
  --image-channel Cellbound2 \
  --n-cells 600 \
  --delta 15 \
  --n-top-genes 25

streamlit run $CSDE_DIR/scripts/annotate.py -- --dir $ANNOTATIONS_PATH
```
