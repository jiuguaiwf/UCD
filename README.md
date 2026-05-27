# Uncertainty-Calibrated Diffusion for Reliable 3D Molecular Graph Generation

[![EDM Branch](https://img.shields.io/badge/Code-EDM%20%2B%20UCD-2ea44f)](edm)
[![RADM Branch](https://img.shields.io/badge/Code-RADM%20%2B%20UCD-f0883e)](radm)

> Fang Wan*, Jingxiang Qu*, Yi Liu  
> State University of New York at Stony Brook  
> KDD 2026  
> *Equal contribution

This repository is the **code release** for **Uncertainty-Calibrated Diffusion for Reliable 3D Molecular Graph Generation**. We study how epistemic uncertainty in the denoiser interacts with diffusion-time noise during reverse sampling, show that this causes systematic variance inflation, and propose **UCD** to calibrate reverse diffusion for more reliable 3D molecule generation.

The release contains two code paths:

- [`edm/`](edm): EDM-based experiments and uncertainty-aware evaluation.
- [`radm/`](radm): RADM-based experiments, uncertainty-aware evaluation, and conditional generation.

## Repository Layout

```text
.
├── edm/    # EDM + UCD
└── radm/   # RADM + UCD
```

## Environment

The two code paths share most dependencies. A simple setup is:

```bash
conda create -n ucd python=3.10
conda activate ucd
conda install -c conda-forge rdkit
pip install torch torchvision numpy scipy matplotlib tqdm wandb imageio timm roma
```

For the original EDM branch, you can also install the baseline requirements directly:

```bash
pip install -r edm/requirements.txt
```

## Data

### QM9

QM9 is downloaded and processed automatically on first use. By default, each codebase stores the processed files under its local `qm9/temp/` directory.

If you already have a processed QM9 cache, you can point the repo to it with:

```bash
export QM9_DATADIR=/path/to/qm9/temp
```

### GEOM-Drugs

Prepare GEOM-Drugs by following the instructions in:

- [`edm/data/geom/README.md`](edm/data/geom/README.md)
- [`radm/data/geom/README.md`](radm/data/geom/README.md)

After preprocessing, place `geom_drugs_30.npy` under each project's `data/geom/` directory or set:

```bash
export GEOM_DRUGS_PATH=/path/to/geom_drugs_30.npy
```

## Intuition

The main paper insight is **variance inflation**: epistemic uncertainty in the denoiser behaves like extra noise during reverse diffusion, so the sampler becomes more stochastic than intended.

At an ideal reverse step, the kernel is

$$
K_t(\mathbf{z}_{t-1}\mid \mathbf{z}_t)
=
\mathcal{N}\!\bigl(\mathbf{z}_{t-1};\, f_t(\mathbf{z}_t),\, \sigma_t^2 I \bigr).
$$

If the denoiser carries epistemic uncertainty with induced variance term $\eta_t^2$, the effective reverse kernel becomes

$$
\tilde K_t(\mathbf{z}_{t-1}\mid \mathbf{z}_t)
=
\mathcal{N}\!\bigl(\mathbf{z}_{t-1};\, f_t(\mathbf{z}_t),\, \tilde{\sigma}_t^2 I \bigr),
\qquad
\tilde{\sigma}_t^2 := \sigma_t^2 + \eta_t^2.
$$

So the issue is not only noisy mean prediction; the **reverse variance itself is inflated** from $\sigma_t^2$ to $\tilde{\sigma}_t^2$. UCD compensates for this by estimating uncertainty online and calibrating the reverse-time noise accordingly. In the EDM branch, this uncertainty estimate is obtained with a **last-layer Laplace approximation**.

## Running UCD

### EDM + UCD on QM9

Train the EDM backbone:

```bash
cd edm
python main_qm9.py \
  --n_epochs 3000 \
  --exp_name edm_qm9 \
  --n_stability_samples 1000 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_steps 1000 \
  --diffusion_loss_type l2 \
  --batch_size 64 \
  --nf 256 \
  --n_layers 9 \
  --lr 1e-4 \
  --normalize_factors [1,4,10] \
  --test_epochs 20 \
  --ema_decay 0.9999
```

Evaluate with UCD using the **last-layer Laplace approximation** for epistemic uncertainty:

```bash
python eval_analyze-uncertainty.py \
  --model_path outputs/edm_qm9_unc \
  --n_samples 10000 \
  --variance_cal_times 20 \
  --dynamic_weights 1 \
  --u_max 0.00037 \
  --uncertainty_method laplace \
  --laplace_n_batches 200
```

### RADM + UCD on QM9

Train the non-equivariant autoencoder:

```bash
cd radm
python qm9_ae.py \
  --n_epochs 200 \
  --batch_size 64 \
  --nf 256 \
  --n_layers 9 \
  --rot_layers 2 \
  --lr 1e-4 \
  --test_epochs 2 \
  --ema_decay 0 \
  --latent_nf 1 \
  --dp False \
  --exp_name qm9_ae \
  --clip_grad False
```

Train the latent diffusion model:

```bash
python qm9_ldm.py \
  --n_epochs 6000 \
  --n_stability_samples 1000 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_steps 1000 \
  --diffusion_loss_type l2 \
  --batch_size 256 \
  --lr 1e-4 \
  --test_epochs 20 \
  --ema_decay 0.9999 \
  --latent_nf 1 \
  --exp_name qm9_ldm_base \
  --dp False \
  --size base \
  --ae_path /path/to/ae
```

Evaluate with uncertainty-aware sampling:

```bash
python eval_analyze-uncertainty.py \
  --model_path outputs/qm9_ldm_base \
  --n_samples 10000 \
  --variance_cal_times 20 \
  --dynamic_weights 1 \
  --u_max 0.00037
```

### Conditional Generation with RADM

Train a conditional RADM model on QM9:

```bash
python qm9_ldm.py \
  --n_epochs 4000 \
  --n_stability_samples 500 \
  --diffusion_noise_schedule polynomial_2 \
  --diffusion_noise_precision 1e-5 \
  --diffusion_steps 1000 \
  --diffusion_loss_type l2 \
  --batch_size 256 \
  --lr 1e-4 \
  --test_epochs 20 \
  --ema_decay 0.9999 \
  --latent_nf 1 \
  --dp False \
  --size base \
  --ae_path /path/to/ae \
  --dataset qm9_second_half \
  --conditioning alpha \
  --exp_name qm9_alpha
```

Supported target properties are `alpha`, `gap`, `homo`, `lumo`, `mu`, and `Cv`.

Train the property predictor:

```bash
cd qm9/property_prediction
python main_qm9_prop.py \
  --num_workers 2 \
  --lr 5e-4 \
  --property alpha \
  --exp_name exp_class_alpha \
  --model_name egnn
```

Evaluate conditional generation:

```bash
cd ../../
python eval_conditional_qm9.py \
  --generators_path outputs/qm9_alpha \
  --classifiers_path qm9/property_prediction/outputs/exp_class_alpha \
  --property alpha \
  --iterations 100 \
  --batch_size 100 \
  --task edm
```

## Acknowledgements

This codebase builds on top of excellent prior work, especially:

- [EDM: E(3) Equivariant Diffusion Model for Molecule Generation in 3D](https://github.com/ehoogeboom/e3_diffusion_for_molecules)
- [GeoLDM](https://github.com/MinkaiXu/GeoLDM)
- [RADM](https://github.com/skeletondyh/RADM)
