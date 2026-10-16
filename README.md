<div align="center">

<h1>Embedding Predictive Adaptation and Refinement for Multi-Modal Industrial Anomaly Detection (EPAR)</h1>

<p>
  <strong><a href="https://github.com/GluteusStrength">Jaeho Cho</a>, <a href="https://jinsikbang.github.io">Jinsik Bang</a>, <a href="https://sites.google.com/view/taehwankim">Taehwan Kim</strong>
</p>

<p>Ulsan National Institute of Science and Technology (UNIST)</p>

<p><em>WACV 2027</em></p>

</div>

---

## Overview

**EPAR** is a two-stage framework for RGB–3D industrial anomaly detection through embedding predictive adaptation and refinement. The first stage adapts RGB and 3D features, and the second stage uses a Noise Suppression Network (NSN) to refine anomaly predictions based on memory-bank retrieval.

### Stage 1: Feature Adaptation

RGB images and 3D point clouds are divided into context and target regions. The context encoders learn to predict target embeddings at multiple feature levels, with exponential moving average (EMA) target encoders and variance–invariance–covariance (VIC) regularization.

<p align="center">
  <a href="figures/feature_adaptation.pdf">
    <img src="figures/feature_adaptation.png" width="100%" alt="Stage 1: RGB and 3D input preparation, multi-level embedding prediction, and VIC regularization.">
  </a>
</p>

*Stage 1: Feature adaptation through multi-level embedding prediction.*

### Stage 2: Anomaly Map Refinement

Pseudo anomalies are synthesized in RGB images and 3D point clouds using foreground-aware masks. With the adapted encoders and memory banks frozen, retrieved features are used to train the NSN against pseudo-anomaly masks, refining anomaly predictions through noise suppression.

<p align="center">
  <a href="figures/refinement_stage.pdf">
    <img src="figures/refinement_stage.png" width="100%" alt="Stage 2: RGB and point-cloud pseudo-anomaly generation followed by Noise Suppression Network training with memory-bank retrieval.">
  </a>
</p>

*Stage 2: Refinement through pseudo-anomaly generation and NSN training.*

## Installation

Run the following commands from the repository root:

```bash
conda create -n mmad python=3.10 pip -y
conda activate mmad
python -m pip install setuptools==58.2.0 wheel==0.45.1
python -m pip install -r requirements.txt

export CUDA_HOME=/usr/local/cuda-11.7
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# Select your GPU architecture: e.g, 8.0 for A100; 8.6 for RTX 3090 / RTX A6000.
export TORCH_CUDA_ARCH_LIST="8.6"

python -m pip install --no-build-isolation ./vendor/pointnet2_ops
FORCE_CUDA=1 python -m pip install --no-build-isolation ./vendor/pytorch3d
python -m pip install --no-build-isolation ./vendor/knn_cuda
```

Build the extensions on a GPU node. If your `mmad` environment already contains these dependencies, activate it directly.

## Data Preparation

### Dataset

Download the datasets from their official pages and extract them into the locations below:

| Dataset | Download | Location in EPAR |
| --- | --- | --- |
| MVTec 3D-AD | [Official dataset and download page](https://www.mvtec.com/research-teaching/datasets/mvtec-3d-ad) | `datasets/mvtec/<category>/` |
| Describable Textures Dataset (DTD) | [Official dataset page](https://www.robots.ox.ac.uk/~vgg/data/dtd/) · [Download releases](https://www.robots.ox.ac.uk/~vgg/data/dtd/releases.html) | `datasets/dtd/images/<texture_category>/` |

Prepare MVTec 3D-AD with its `train` and `test` splits under each category directory. Evaluation also requires the training split for memory bank construction. Training also requires the `validation` split.

Arrange the datasets under `EPAR/datasets/` as follows, matching the AD-JEPA dataset layout:

```text
EPAR/
└── datasets/
    ├── mvtec/
    │   ├── bagel/
    │   │   ├── train/good/
    │   │   │   ├── rgb/                 # *.png
    │   │   │   └── xyz/                 # *.tiff
    │   │   ├── validation/good/
    │   │   │   ├── rgb/
    │   │   │   └── xyz/
    │   │   └── test/
    │   │       ├── good/
    │   │       │   ├── rgb/
    │   │       │   └── xyz/
    │   │       └── <defect_type>/       # e.g., crack, hole
    │   │           ├── rgb/
    │   │           ├── xyz/
    │   │           └── gt/             # anomaly masks (*.png)
    │   └── <other_categories>/         # same split structure
    └── dtd/
        ├── images/
        │   └── <texture_category>/    # e.g., banded, cracked
        │       └── *.jpg
        ├── labels/
        └── imdb/
```

The ten MVTec categories are `bagel`, `cable_gland`, `carrot`, `cookie`, `dowel`, `foam`, `peach`, `potato`, `rope`, and `tire`. Preserve any additional files and folders supplied with the dataset, such as calibration files and masks for normal test samples.

Extract Describable Textures Dataset (DTD) so that texture images are at `datasets/dtd/images/<texture_category>/*.jpg`. DTD provides textures for anomaly synthesis; it is separate from MVTec and is not processed by the MVTec preprocessing script. The loader reads `images/`; retain the archive's `labels/` and `imdb/` folders as supplied.

With this layout, keep `paths.dtd_dataset_path: null` in `config/mvtec3d.yaml`: EPAR automatically uses `datasets/dtd/` beside `datasets/mvtec/`. For a different location, set `paths.dtd_dataset_path` to the DTD root containing `images/`, for example `datasets/dtd` or an absolute path (not the `images/` directory itself).

Following M3DM, we preprocess MVTec 3D-AD by removing the background plane with RANSAC, padding RGB images, XYZ point clouds, and available ground-truth masks to square dimensions, and cleaning point-cloud components with DBSCAN. Use [`utils/preprocess_mvtec.py`](utils/preprocess_mvtec.py) to run this preprocessing.

Run the following commands from the repository root, passing the directory that directly contains the category folders (e.g., `bagel/`, `cable_gland/`):

```bash
python -m pip install open3d==0.18.0
python utils/preprocess_mvtec.py datasets/mvtec
```

The script processes the `train`, `validation`, and `test` splits **in place**, overwriting RGB images, XYZ TIFF files, and available ground-truth masks. Keep a copy of the raw dataset and run preprocessing only once on the working copy.

For training and inference, `--dataset_path` may point to the preprocessed `mvtec/` directory or its parent.

## Training Configuration

All training hyperparameters and class selections are defined in [`config/mvtec3d.yaml`](config/mvtec3d.yaml). Shared settings are in `common`, stage-specific settings are in `stages`, and per-class settings are in `class_overrides` (priority: `common` < `stages` < `class_overrides`). To run a single class, set `classes: [bagel]`.

Set the DINO ViT-B/8 and Point-MAE checkpoint paths in [`config/mvtec3d.yaml`](config/mvtec3d.yaml):

```yaml
paths:
  visual_encoder_ckpt_pth: checkpoints/pretrained/vit_base_patch8_224_dino.pth
  point_encoder_ckpt_pth: checkpoints/pretrained/pointmae_pretrain.pth
```

Update these entries in the existing YAML; retain the other settings. Relative paths are resolved from the repository root. Set `paths.dtd_dataset_path` if DTD textures are stored elsewhere; the default is `dtd/` alongside `mvtec/`.

## Training and Inference

Run the stages in order:

```bash
python train_1st.py --dataset_path datasets/mvtec
python train_2nd.py --dataset_path datasets/mvtec
python inference.py --dataset_path datasets/mvtec
```

The only CLI argument for these stages is `--dataset_path`; configure the remaining settings in the YAML file above.

Checkpoints are saved under `checkpoints/stage1/` and `checkpoints/stage2/`. Inference metrics are written to `outputs/inference/<class>/run.log`; training uses console output only.

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgements

Our work builds on [I-JEPA (CVPR 2023)](https://github.com/facebookresearch/ijepa), [Point-JEPA (WACV 2025)](https://github.com/Ayumu-J-S/Point-JEPA), [M3DM (CVPR 2023)](https://github.com/wynoxos/M3DM), and [VIC-Regularization (VICReg, ICLR 2022)](https://github.com/facebookresearch/vicreg). We thank the authors for sharing their code and research.
