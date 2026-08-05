# scvi_neural_ode

## Overview
This package implements the odescvi model (see citation below), which computes pseudotime trajectories from scRNA-seq data. The model builds up on the scvi framework (Lopez et al, Deep generative modeling for single-cell transcriptomics. Nat Methods 15, 1053–1058 (2018)) - it computes a low dimensional latent representation from gene expression data via a Variational Autoencoder (VAE; Kingma, D. P. & Welling, M. Auto-Encoding Variational Bayes. arXiv (2013), and fits continuous trajectories in this latent space via neural ordinary differential equations (Chen et al, Neural Ordinary Differential Equations. Proceedings of the 32nd International Conference on Neural Information Processing Systems. 6572-6583). The ordering of the cells on this trajectory is quantified via pseudotime, a real number between 0 and 1 assigned to each cell. The repo structure is based on the [scvi-tools-skeleton](https://github.com/scverse/scvi-tools-skeleton) template.

The package allows distinct modeling frameworks when working with datasets of multiple cell populations:
  1. An individual trajectory for each population - e.g. in cases where the dataset is made up of distinct cell types, each following their own development.
  2. One trajectory across all populations - e.g. when populations are experiment batches of the same cell type and we'd like integrate all cells into one trajectory describing the development.
  3. Multiple trajectories with a known root population - e.g. when we have a progenitor population that develops into multiple cell types.

## System Requirements
### Hardware Requirements
The package requires a standard computer, with minimum RAM of 2GB. For optimal performance we recommend 16+ GB RAM and CPU clock speed of 3.2+ GHZ.

### Software Requirements
The package was tested on Ubuntu 20.04.6 LTS and Python version 3.12.

The Python package requirements are listed in the `pyproject.toml` file under `[project]`. The package was tested in an environment with the following package versions:
```
numpy==2.4.2
pandas==2.3.3
scikit-learn==1.8.0
anndata==0.12.10
scanpy==1.12
torch==2.10.0
lightning==2.6.1
torchdiffeq==0.2.5
```

## Installation
We recommend installing the package in a new environment. Run the following command in your environment:
```
pip install git+https://github.com/AtanasiuDemian/scvi_neural_ode.git
```
Or install a specific release:
```
pip install git+https://github.com/AtanasiuDemian/scvi_neural_ode.git@v1.0.0
```

For developing the package:
```
git clone +https://github.com/AtanasiuDemian/scvi_neural_ode.git
cd scvi_neural_ode
pip install -e ".[dev]"
```
Installation should only take a few minutes.

## Instructions for use
### Data Format
The model expects a dataset in `anndata` format as input, where samples are cells and features are genes, with metadata information stored in the `.obs` attribute. We focus in particular on the batch/label column which will be used as input to the conditional VAE. This column is formatted as integers ranging from 0 to `n_categories-1`. In this tutorial we work with data object `adata` and we call this column `label`. To be used by the VAE, it needs to be set up via the `setup_data_registry` function:
```py
from scvi_neural_ode.data import setup_data_registry
setup_data_registry(adata=adata, batch_key='label')
```
The VAE models can be found in the `models/` folder; there are 3 classes, depending on the use cases described above:
### Individual trajectory for each population
Use the [`ODESCVI`](https://github.com/AtanasiuDemian/scvi_neural_ode/blob/main/src/scvi_neural_ode/models/odescvi.py) class. In this example we define a latent space of dimension 10, 1 hidden node in both encoder & decoder, each with 128 nodes, dropout rate of 0.1, and batch size of 128 cells. 

```py
from scvi_neural_ode.models import ODESCVI

model = ODESCVI(adata=adata, n_latent=10, n_hidden=128, n_layers_encoder=1, n_layers_decoder=1, dropout_rate=0.1, batch_size=128)
model.train(n_epochs=200, lr=5e-3, accelerator='auto')

# Generate the latent representation (returns numpy array)
latents = model.get_latent_representation()

# Generate the reconstructed gene proportions (returns numpy array)
expr = model.get_normalized_expression()

# Generate the pseudotime values (returns numpy array)
pseudotime = model.get_time()

# Generate all trajectory output - returns a dictionary made up of a dictionary for each label.
# Each such dictionary contains pseudotime (T), predicted trajectory (pred_z), decoded trajectory (pred_x), latents (encoder output: z_mean, z_var, z_sample)
# Note that this output is sorted through pseudotime!
traj_output = model.get_trajectory_output()
```
### One trajectory across all populations
Use [CondODESCVI](https://github.com/AtanasiuDemian/scvi_neural_ode/blob/main/src/scvi_neural_ode/models/condodescvi.py) class. Class instantiation and model training is the same as above, methods `get_latent_representation`, `get_normalized_expression`, `get_time` work as above. 

```py
# To get trajectory output use forward_pass
# Here only pred_z (predicted latent trajectory) and T (pseudotime) will be sorted.
traj_output = model.forward_pass()
```

### Multiple trajectories with known root population
Use [BranchingCondODESCVI](https://github.com/AtanasiuDemian/scvi_neural_ode/blob/main/src/scvi_neural_ode/models/branch_condodescvi.py). Here the model combines aspects from the other 2 use cases: we can compute multiple trajectories, one for each population, and each such trajectory can integrate multiple subpopulations. E.g. if we have multiple cell types, with one progenitor which all other populations develop from, and the dataset contains multiple experiment batches that we'd like to integrate into one representation. In this case we use the `label` column to describe the cell types - category 0 is assumed to be the root population. We use another column `batch` for batch information. We use `setup_data_registry` only for covariates that we integrate out i.e. batch information, while `label` will be called in the class instantiation:
```py
from scvi_neural_ode.data import setup_data_registry
from scvi_neural_ode.models import BranchingCondODESCVI

setup_data_registry(adata=adata, batch_key='batch')

model = BranchingCondODESCVI(adata=adata, n_cats=4, CAT_KEY='label') # n_cats = number of cell types.
model.train(n_epochs=200, lr=5e-3, accelerator='auto')
```
We can also pinpoint a specific cell as root, as long as it's part of label category 0 - use the `iroot` argument in class instantiation above, this is the index of the cell in the `anndata` object.

Methods `get_latent_representation` and `get_normalized_expression` work as in the other classes, and `forward_pass` like in `CondODESCVI`. Use this latter method to get pseudotime. 

## Citation
If you use this in your work, please cite:

Atanasiu Stefan Demian, Giorgio Anselmi, Elitza Deltcheva, Naeema Mehmood, Matthew Nicholls, Jason Wray, Tariq Enver, Marella F.T.R de Bruijn, Edward Morrissey. "**NAVI: a variational autoencoder model for neighbourhood analysis using pairs of physically interacting cells**" (2026), _submitted_
