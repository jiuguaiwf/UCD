# clock_test.py (radm-style, robust for latent diffusion + edge_mask shape)
# Loads args.pickle + generative_model(_ema).npy using get_latent_diffusion exactly like eval_analyze.py
# Times:
#   sample_p_zs_given_zt
#   sample_p_zs_given_zt_uncertainty
# Uses RANDOM real batches built from dataset + collate_fn.
#
# Fixes included:
#  (1) Build xh to match eps.shape[-1] (latent diffusion state dim), not (3+onehot+charges).
#  (2) Ensure edge_mask is [B, N, N] because en_diffusion.phi() adds an eye(N) to it.

# Rdkit import should be first, do not move it
try:
    from rdkit import Chem  # noqa: F401
except ModuleNotFoundError:
    pass

import argparse
import pickle
import time
from os.path import join

import torch

import utils
from qm9 import dataset
from qm9.models import get_latent_diffusion
from qm9.utils import prepare_context, compute_mean_mad
from configs.datasets_config import get_dataset_info
from equivariant_diffusion.utils import remove_mean_with_mask


def _unwrap(m):
    return m.module if hasattr(m, "module") else m


def _safe_setdefault(args, name, value):
    if not hasattr(args, name):
        setattr(args, name, value)


def _to_device(batch, device, dtype):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            if v.dtype.is_floating_point:
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


def _ensure_edge_mask_bnn(edge_mask: torch.Tensor, bs: int, n_nodes: int) -> torch.Tensor:
    """
    en_diffusion.phi() expects edge_mask shaped [B, N, N] (float/bool ok).
    Many dataloaders store flattened [B*N*N, 1] or [B, N*N].
    Convert robustly.
    """
    # [B,1,N,N] -> [B,N,N]
    if edge_mask.dim() == 4 and edge_mask.size(1) == 1:
        return edge_mask[:, 0]

    # already [B,N,N]
    if edge_mask.dim() == 3:
        return edge_mask

    # [B*N*N, 1] -> [B,N,N]
    if edge_mask.dim() == 2:
        if edge_mask.size(-1) == 1 and edge_mask.size(0) == bs * n_nodes * n_nodes:
            return edge_mask.view(bs, n_nodes, n_nodes)
        # [B, N*N] -> [B,N,N]
        if edge_mask.size(0) == bs and edge_mask.size(1) == n_nodes * n_nodes:
            return edge_mask.view(bs, n_nodes, n_nodes)

    # [B*N*N] -> [B,N,N]
    if edge_mask.dim() == 1 and edge_mask.numel() == bs * n_nodes * n_nodes:
        return edge_mask.view(bs, n_nodes, n_nodes)

    raise RuntimeError(
        f"Unsupported edge_mask shape: {tuple(edge_mask.shape)} (bs={bs}, n_nodes={n_nodes})"
    )


@torch.no_grad()
def time_two_kernels(args, eval_args, device, dtype, generative_model, batch, property_norms):
    """
    Build z_t like training, then time:
      sample_p_zs_given_zt
      sample_p_zs_given_zt_uncertainty
    """
    m = _unwrap(generative_model)
    m.eval()

    x = batch["positions"].to(device, dtype)
    node_mask = batch["atom_mask"].to(device, dtype).unsqueeze(2)  # [B,N,1]
    one_hot = batch["one_hot"].to(device, dtype)

    bs, n_nodes, _ = x.shape

    # edge_mask MUST be [B,N,N] for en_diffusion.phi()
    edge_mask = batch["edge_mask"].to(device, dtype)
    edge_mask = _ensure_edge_mask_bnn(edge_mask, bs, n_nodes)

    # charges
    if args.include_charges:
        charges = batch["charges"].to(device, dtype)
        if charges.dim() == 2:
            charges = charges.unsqueeze(-1)
    else:
        charges = torch.zeros(bs, n_nodes, 0, device=device, dtype=dtype)

    # training preprocessing
    x = remove_mean_with_mask(x, node_mask)

    # context (same logic as training/eval)
    if len(args.conditioning) > 0:
        context = prepare_context(args.conditioning, batch, property_norms).to(device, dtype)
    else:
        context = None

    # choose mid step
    t_int_val = float(args.diffusion_steps // 2)
    if t_int_val < 1:
        t_int_val = 1.0

    t_int = torch.full((bs, 1), t_int_val, device=device, dtype=torch.float32)
    s_int = t_int - 1.0

    T = float(getattr(m, "T", args.diffusion_steps))
    t = t_int / T
    s = s_int / T

    # eps defines diffusion state dimension (latent diffusion may be 3+latent_dim)
    eps = m.sample_combined_position_feature_noise(
        n_samples=bs, n_nodes=n_nodes, node_mask=node_mask
    )  # [B,N,D_state]

    # Build xh in SAME space as eps
    xh = torch.zeros_like(eps)
    xh[:, :, :3] = x
    feat_dim = eps.shape[-1] - 3
    if feat_dim > 0:
        raw_feat = torch.cat([one_hot, charges], dim=2)  # [B,N,K]
        if raw_feat.shape[-1] >= feat_dim:
            xh[:, :, 3:] = raw_feat[:, :, :feat_dim]
        else:
            xh[:, :, 3:3 + raw_feat.shape[-1]] = raw_feat

    # gamma/alpha/sigma inflated to xh shape
    gamma_t = m.inflate_batch_array(m.gamma(t), xh)
    alpha_t = m.alpha(gamma_t, xh)
    sigma_t = m.sigma(gamma_t, xh)

    z_t = alpha_t * xh + sigma_t * eps

    mc_times = int(eval_args.profile_mc_times)
    n_iters = int(eval_args.profile_n_iters)

    # warmup
    for _ in range(5):
        _ = m.sample_p_zs_given_zt(s, t, z_t, node_mask, edge_mask, context)
        _ = m.sample_p_zs_given_zt_uncertainty(
            s, t, z_t, node_mask, edge_mask, context,
            variance_cal_times=mc_times
        )

    # time normal
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        _ = m.sample_p_zs_given_zt(s, t, z_t, node_mask, edge_mask, context)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    avg_normal = (t1 - t0) / n_iters

    # time uncertainty
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        _ = m.sample_p_zs_given_zt_uncertainty(
            s, t, z_t, node_mask, edge_mask, context,
            variance_cal_times=mc_times
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    avg_unc = (t1 - t0) / n_iters

    return avg_normal, avg_unc, bs, n_nodes, int(t_int_val), int(eps.shape[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Folder with args.pickle + generative_model*.npy")
    parser.add_argument("--profile_mc_times", type=int, default=20)
    parser.add_argument("--profile_n_iters", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=100, help="Timing batch size (random batch sampling)")
    parser.add_argument("--use_ema", type=int, default=1, help="1=generative_model_ema.npy if ema_decay>0 else generative_model.npy")
    eval_args, _ = parser.parse_known_args()

    torch.manual_seed(eval_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_args.seed)

    # ---- load args.pickle exactly like eval_analyze.py ----
    with open(join(eval_args.model_path, "args.pickle"), "rb") as f:
        args = pickle.load(f)

    # minimal defaults to avoid missing attrs
    _safe_setdefault(args, "normalization_factor", 1.0)
    _safe_setdefault(args, "aggregation_method", "sum")
    _safe_setdefault(args, "num_workers", 0)
    _safe_setdefault(args, "filter_n_atoms", None)
    _safe_setdefault(args, "sequential", False)
    _safe_setdefault(args, "include_charges", True)
    _safe_setdefault(args, "conditioning", [])
    _safe_setdefault(args, "ema_decay", 0.0)
    _safe_setdefault(args, "diffusion_steps", 1000)

    # override batch size for timing only
    args.batch_size = int(eval_args.batch_size)

    # device
    args.cuda = (not getattr(args, "no_cuda", False)) and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32
    args.device = device

    # keep consistent with repo style
    utils.create_folders(args)

    # dataloaders
    dataloaders, _charge_scale = dataset.retrieve_dataloaders(args)
    train_loader = dataloaders["train"]

    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    # model (MATCH eval_analyze.py)
    generative_model, _nodes_dist, prop_dist = get_latent_diffusion(
        args, device, dataset_info, train_loader
    )

    # property norms (only if conditioning)
    if prop_dist is not None and len(args.conditioning) > 0:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)
    else:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset) if len(args.conditioning) > 0 else None

    generative_model.to(device)

    # load weights (MATCH eval_analyze.py)
    fn = "generative_model_ema.npy" if (eval_args.use_ema and getattr(args, "ema_decay", 0.0) > 0) else "generative_model.npy"
    weight_path = join(eval_args.model_path, fn)
    state = torch.load(weight_path, map_location=device)
    generative_model.load_state_dict(state)
    print(f"[OK] Loaded: {weight_path}")

    # random batches from dataset + collate_fn
    train_ds = train_loader.dataset
    collate_fn = getattr(train_loader, "collate_fn", None)
    if collate_fn is None:
        raise RuntimeError("train_loader has no collate_fn; cannot build random batches safely.")

    N = len(train_ds)
    bs = int(eval_args.batch_size)

    normal_times = []
    unc_times = []

    for k in range(int(eval_args.iterations)):
        idx = torch.randint(low=0, high=N, size=(bs,)).tolist()
        samples = [train_ds[i] for i in idx]
        batch = collate_fn(samples)
        batch = _to_device(batch, device=device, dtype=dtype)

        avg_normal, avg_unc, bsz, n_nodes, tmid, dstate = time_two_kernels(
            args=args,
            eval_args=eval_args,
            device=device,
            dtype=dtype,
            generative_model=generative_model,
            batch=batch,
            property_norms=property_norms,
        )

        normal_times.append(avg_normal)
        unc_times.append(avg_unc)

        print(
            f"[ITER {k+1}/{eval_args.iterations}] bs={bsz}, N={n_nodes}, D_state={dstate}, t_mid={tmid} | "
            f"p(zs|zt): {avg_normal*1000:.2f} ms | "
            f"p(zs|zt,unc): {avg_unc*1000:.2f} ms"
        )

    nt = torch.tensor(normal_times)
    ut = torch.tensor(unc_times)

    print("\n[SUMMARY]")
    print(f" random batches, iterations={eval_args.iterations}, batch_size={bs}, seed={eval_args.seed}")
    print(f" sample_p_zs_given_zt             mean {nt.mean().item()*1000:.2f} ms  std {nt.std(unbiased=False).item()*1000:.2f} ms")
    print(f" sample_p_zs_given_zt_uncertainty mean {ut.mean().item()*1000:.2f} ms  std {ut.std(unbiased=False).item()*1000:.2f} ms")
    print(f" MC={int(eval_args.profile_mc_times)}  inner iters={int(eval_args.profile_n_iters)}\n")


if __name__ == "__main__":
    main()
