<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p> 

### DRESS: Disentangled Representation-based Self-Supervised Meta-Learning for Diverse Tasks [[arXiv]](https://arxiv.org/abs/2503.09679)
Authors: Wei Cui, Tongzi Wu, Jesse C. Cresswell, Yi Sui, Keyvan Golestan


## Summary
This repository contains the official implementation of the paper <em>DRESS: Disentangled Representation-based Self-Supervised Meta-Learning for Diverse Tasks</em>. It includes both training and evaluation code.

## Repository Structure
The code files within the repository are organized as follows:
* `main.py`: the main entrance point of the program.
* `partition_generators.py`: implementation of generating supervised and self-supervised partitions on each dataset.
* `task_generator.py`: implementation of generating few-shot learning tasks from any given partition.
* `utils.py`: implementation of helper functions.

The sub-folders within the repository are as follows:
* `scripts/`: the folder including the scripts to train, evaluate, and obtain visulizations.
* `encoders/`: the folder containing classes of encoders for obtaining the latent spaces.
* `dataset_loaders/`: the folder containing scripts for loading each of the dataset for experiments.
* `baselines/`: the folder containing implementations of baseline methods.
* `analyze_results/`: the folder containing scripts for post-processing results.
* `visualization_results/`: the folder containing visualizations on constructed tasks via DRESS.

## Dataset
Create a folder named `data/` under the main directory to house the raw data.
The datasets experimented are loaded from their respective dataset loader script under `dataset_loaders/`. The source data preparations are as follows:
* smallNORB: automatically downloaded within our script via the `tensorflow_datasets` package.
* shapes3D: download `3dshapes.h5` from [Google Cloud Storage](https://console.cloud.google.com/storage/browser/3d-shapes) and place it under `data/shapes3d/`.
* causal3D: download `trainset.tar.gz` and `testset.tar.gz` from the [dataset homepage](https://zenodo.org/records/4784282#.YgWo0PXMKbg) and extract them under `data/causal3d/train/` and `data/causal3d/test/` resectively.
* MPI3D: download `mpi3d_toy.npz` from this [link](https://storage.googleapis.com/mpi3d_disentanglement_dataset/data/mpi3d_toy.npz) and place it under `data/mpi3d/`.
* CelebA: automatically downloaded within our script via the `torchvision` package.

## Running Environment
Simply install an anaconda environment using the `environment.yml` file under this repository.
