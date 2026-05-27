# # clock_test.py  (radm-style version: LOAD args.pickle + model weights from --model_path)
# # Times:
# #   sample_p_zs_given_zt
# #   sample_p_zs_given_zt_uncertainty
# #
# # Uses REAL data batch from dataloaders['train'] with training masks/shapes.

# # Rdkit import should be first, do not move it
# try:
#     from rdkit import Chem  # noqa: F401
# except ModuleNotFoundError:
#     pass

# import argparse
# import pickle
# import time
# from os.path import join

# import torch

# import utils
# from configs.datasets_config import get_dataset_info
# from qm9 import dataset
# from qm9.models import get_model  # this returns EnVariationalDiffusion for diffusion models
# from qm9.utils import prepare_context, compute_mean_mad
# from equivariant_diffusion.utils import remove_mean_with_mask


# def _unwrap(m):
#     return m.module if hasattr(m, "module") else m


# def _safe_setdefault(args, name, value):
#     if not hasattr(args, name):
#         setattr(args, name, value)


# @torch.no_grad()
# def run_timing_once(args, eval_args, device, dtype, model, loader, property_norms):
#     m = _unwrap(model)
#     m.eval()

#     data = next(iter(loader))

#     x = data["positions"].to(device, dtype)
#     node_mask = data["atom_mask"].to(device, dtype).unsqueeze(2)
#     edge_mask = data["edge_mask"].to(device, dtype)
#     one_hot = data["one_hot"].to(device, dtype)

#     bs, n_nodes, _ = x.shape

#     if args.include_charges:
#         charges = data["charges"].to(device, dtype)
#         if charges.dim() == 2:
#             charges = charges.unsqueeze(-1)
#     else:
#         charges = torch.zeros(bs, n_nodes, 0, device=device, dtype=dtype)

#     # match training preprocessing
#     x = remove_mean_with_mask(x, node_mask)

#     h = {"categorical": one_hot, "integer": charges}

#     # context exactly like training/eval
#     if len(args.conditioning) > 0:
#         # compute_mean_mad was used to create property_norms; prepare_context uses it
#         context = prepare_context(args.conditioning, data, property_norms).to(device, dtype)
#     else:
#         context = None

#     # ----------------------------
#     # DIRECTLY CREATE z_t (EDM training formula)
#     # z_t = alpha_t * xh + sigma_t * eps
#     # where xh = cat([x, categorical, integer])
#     # and t is continuous normalized in [0,1], s = t - 1/T
#     # ----------------------------
#     # choose mid step
#     t_int_val = float(args.diffusion_steps // 2)
#     if t_int_val < 1:
#         t_int_val = 1.0

#     # t_int, s_int shape [bs, 1] to match EDM codepaths
#     t_int = torch.full((bs, 1), t_int_val, device=device, dtype=torch.float32)
#     s_int = t_int - 1.0

#     # normalized time (EnVariationalDiffusion usually stores T)
#     T = float(getattr(m, "T", args.diffusion_steps))
#     t = t_int / T
#     s = s_int / T

#     gamma_t = m.inflate_batch_array(m.gamma(t), x)
#     alpha_t = m.alpha(gamma_t, x)
#     sigma_t = m.sigma(gamma_t, x)

#     eps = m.sample_combined_position_feature_noise(
#         n_samples=bs,
#         n_nodes=n_nodes,
#         node_mask=node_mask
#     )

#     xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
#     z_t = alpha_t * xh + sigma_t * eps

#     mc_times = int(eval_args.profile_mc_times)
#     n_iters = int(eval_args.profile_n_iters)

#     # warmup
#     for _ in range(5):
#         _ = m.sample_p_zs_given_zt(s, t, z_t, node_mask, edge_mask, context)
#         _ = m.sample_p_zs_given_zt_uncertainty(
#             s, t, z_t, node_mask, edge_mask, context,
#             variance_cal_times=mc_times
#         )

#     # time normal
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     t0 = time.time()
#     for _ in range(n_iters):
#         _ = m.sample_p_zs_given_zt(s, t, z_t, node_mask, edge_mask, context)
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     t1 = time.time()
#     avg_normal = (t1 - t0) / n_iters

#     # time uncertainty
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     t0 = time.time()
#     for _ in range(n_iters):
#         _ = m.sample_p_zs_given_zt_uncertainty(
#             s, t, z_t, node_mask, edge_mask, context,
#             variance_cal_times=mc_times
#         )
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#     t1 = time.time()
#     avg_unc = (t1 - t0) / n_iters

#     print(
#         "\n[TIMING RESULTS]"
#         f"\n model_path = {eval_args.model_path}"
#         f"\n batch_size = {bs}"
#         f"\n max_n_nodes = {n_nodes}"
#         f"\n t_int = {int(t_int[0].item())} (t={t[0].item():.4f}, s={s[0].item():.4f})"
#         f"\n sample_p_zs_given_zt              : {avg_normal*1000:.2f} ms"
#         f"\n sample_p_zs_given_zt_uncertainty  : {avg_unc*1000:.2f} ms"
#         f"\n   MC = {mc_times}, per-pass = {avg_unc*1000/mc_times:.2f} ms\n"
#     )


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, required=True, help="Path containing args.pickle + model weights")
#     parser.add_argument("--profile_mc_times", type=int, default=20)
#     parser.add_argument("--profile_n_iters", type=int, default=20)
#     parser.add_argument("--use_ema", type=int, default=1, help="1: load *_ema.npy if exists; 0: load non-ema")
#     eval_args, _ = parser.parse_known_args()

#     # ---- load training args exactly like your eval script ----
#     with open(join(eval_args.model_path, "args.pickle"), "rb") as f:
#         args = pickle.load(f)

#     # ---- fill any missing defaults safely ----
#     _safe_setdefault(args, "probabilistic_model", "diffusion")
#     _safe_setdefault(args, "normalization_factor", 1.0)
#     _safe_setdefault(args, "aggregation_method", "sum")
#     _safe_setdefault(args, "num_workers", 0)
#     _safe_setdefault(args, "filter_n_atoms", None)
#     _safe_setdefault(args, "sequential", False)
#     _safe_setdefault(args, "include_charges", True)
#     _safe_setdefault(args, "conditioning", [])

#     # device
#     args.cuda = (not getattr(args, "no_cuda", False)) and torch.cuda.is_available()
#     device = torch.device("cuda" if args.cuda else "cpu")
#     dtype = torch.float32

#     # dataloaders
#     dataloaders, _charge_scale = dataset.retrieve_dataloaders(args)

#     dataset_info = get_dataset_info(args.dataset, args.remove_h)

#     # property norms for context, if needed
#     if len(args.conditioning) > 0:
#         property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
#         # get_model needs context_node_nf set
#         data_dummy = next(iter(dataloaders["train"]))
#         context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
#         args.context_node_nf = context_dummy.size(2)
#     else:
#         property_norms = None
#         args.context_node_nf = 0

#     # build model architecture
#     generative_model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders["train"])
#     generative_model.to(device)

#     # load weights (ema or not)
#     # common filenames in these repos:
#     #   generative_model_ema.npy / generative_model.npy
#     #   flow_ema.npy / flow.npy  (older)
#     candidates = []
#     if eval_args.use_ema:
#         candidates += ["generative_model_ema.npy", "flow_ema.npy", "model_ema.npy"]
#     candidates += ["generative_model.npy", "flow.npy", "model.npy"]

#     state_dict = None
#     loaded_name = None
#     for fn in candidates:
#         p = join(eval_args.model_path, fn)
#         try:
#             state_dict = torch.load(p, map_location=device)
#             loaded_name = fn
#             break
#         except FileNotFoundError:
#             continue

#     if state_dict is None:
#         raise FileNotFoundError(
#             f"Could not find model weights in {eval_args.model_path}. Tried: {candidates}"
#         )

#     generative_model.load_state_dict(state_dict)
#     print(f"[OK] Loaded weights: {loaded_name}")

#     run_timing_once(
#         args=args,
#         eval_args=eval_args,
#         device=device,
#         dtype=dtype,
#         model=generative_model,
#         loader=dataloaders["train"],
#         property_norms=property_norms,
#     )


# if __name__ == "__main__":
#     main()

# clock_test.py  (LOAD args.pickle + model weights from --model_path)
# Random batches each iteration, independent of DataLoader shuffle.
# Times:
#   sample_p_zs_given_zt
#   sample_p_zs_given_zt_uncertainty

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
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.models import get_model
from qm9.utils import prepare_context, compute_mean_mad
from equivariant_diffusion.utils import remove_mean_with_mask


def _unwrap(m):
    return m.module if hasattr(m, "module") else m


def _safe_setdefault(args, name, value):
    if not hasattr(args, name):
        setattr(args, name, value)


def _to_device(batch, device, dtype):
    # batch is a dict of tensors
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            # edge_mask / atom_mask sometimes are float/bool already; keep dtype for floats
            if v.dtype.is_floating_point:
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def run_timing_on_batch(args, eval_args, device, dtype, model, data, property_norms):
    m = _unwrap(model)
    m.eval()

    x = data["positions"].to(device, dtype)
    node_mask = data["atom_mask"].to(device, dtype).unsqueeze(2)
    edge_mask = data["edge_mask"].to(device, dtype)
    one_hot = data["one_hot"].to(device, dtype)

    bs, n_nodes, _ = x.shape

    if args.include_charges:
        charges = data["charges"].to(device, dtype)
        if charges.dim() == 2:
            charges = charges.unsqueeze(-1)
    else:
        charges = torch.zeros(bs, n_nodes, 0, device=device, dtype=dtype)

    # match training preprocessing
    x = remove_mean_with_mask(x, node_mask)

    h = {"categorical": one_hot, "integer": charges}

    # context exactly like training/eval
    if len(args.conditioning) > 0:
        context = prepare_context(args.conditioning, data, property_norms).to(device, dtype)
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

    gamma_t = m.inflate_batch_array(m.gamma(t), x)
    alpha_t = m.alpha(gamma_t, x)
    sigma_t = m.sigma(gamma_t, x)

    eps = m.sample_combined_position_feature_noise(
        n_samples=bs,
        n_nodes=n_nodes,
        node_mask=node_mask
    )

    xh = torch.cat([x, h["categorical"], h["integer"]], dim=2)
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

    return avg_normal, avg_unc, bs, n_nodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path containing args.pickle + model weights")
    parser.add_argument("--profile_mc_times", type=int, default=20)
    parser.add_argument("--profile_n_iters", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=5, help="How many random batches to time")
    parser.add_argument("--use_ema", type=int, default=1, help="1: load *_ema.npy if exists; 0: load non-ema")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for batch sampling")
    eval_args, _ = parser.parse_known_args()

    torch.manual_seed(eval_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_args.seed)

    # ---- load training args exactly like your eval script ----
    with open(join(eval_args.model_path, "args.pickle"), "rb") as f:
        args = pickle.load(f)

    # ---- fill any missing defaults safely ----
    _safe_setdefault(args, "probabilistic_model", "diffusion")
    _safe_setdefault(args, "normalization_factor", 1.0)
    _safe_setdefault(args, "aggregation_method", "sum")
    _safe_setdefault(args, "num_workers", 0)
    _safe_setdefault(args, "filter_n_atoms", None)
    _safe_setdefault(args, "sequential", False)
    _safe_setdefault(args, "include_charges", True)
    _safe_setdefault(args, "conditioning", [])

    # device
    args.cuda = (not getattr(args, "no_cuda", False)) and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32

    # dataloaders
    dataloaders, _charge_scale = dataset.retrieve_dataloaders(args)
    train_loader = dataloaders["train"]

    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    # property norms for context, if needed
    if len(args.conditioning) > 0:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        data_dummy = next(iter(train_loader))
        context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
        args.context_node_nf = context_dummy.size(2)
    else:
        property_norms = None
        args.context_node_nf = 0

    # build model architecture
    generative_model, _nodes_dist, _prop_dist = get_model(args, device, dataset_info, train_loader)
    generative_model.to(device)

    # load weights (ema or not)
    candidates = []
    if eval_args.use_ema:
        candidates += ["generative_model_ema.npy", "flow_ema.npy", "model_ema.npy"]
    candidates += ["generative_model.npy", "flow.npy", "model.npy"]

    state_dict = None
    loaded_name = None
    for fn in candidates:
        p = join(eval_args.model_path, fn)
        try:
            state_dict = torch.load(p, map_location=device)
            loaded_name = fn
            break
        except FileNotFoundError:
            continue

    if state_dict is None:
        raise FileNotFoundError(f"Could not find model weights in {eval_args.model_path}. Tried: {candidates}")

    generative_model.load_state_dict(state_dict)
    print(f"[OK] Loaded weights: {loaded_name}")

    # ---- TRUE random batching from dataset + collate_fn ----
    train_ds = train_loader.dataset
    collate_fn = getattr(train_loader, "collate_fn", None)
    if collate_fn is None:
        raise RuntimeError("train_loader has no collate_fn; cannot build random batches safely.")

    N = len(train_ds)
    bs = int(getattr(args, "batch_size", 64))

    normal_times = []
    unc_times = []

    for k in range(int(eval_args.iterations)):
        idx = torch.randint(low=0, high=N, size=(bs,)).tolist()
        samples = [train_ds[i] for i in idx]
        batch = collate_fn(samples)  # should match training batch dict
        batch = _to_device(batch, device=device, dtype=dtype)

        avg_normal, avg_unc, bsz, n_nodes = run_timing_on_batch(
            args=args,
            eval_args=eval_args,
            device=device,
            dtype=dtype,
            model=generative_model,
            data=batch,
            property_norms=property_norms,
        )

        normal_times.append(avg_normal)
        unc_times.append(avg_unc)

        print(
            f"[ITER {k+1}/{eval_args.iterations}] bs={bsz}, N={n_nodes} | "
            f"p(zs|zt): {avg_normal*1000:.2f} ms | "
            f"p(zs|zt,unc): {avg_unc*1000:.2f} ms"
        )

    nt = torch.tensor(normal_times)
    ut = torch.tensor(unc_times)

    print("\n[SUMMARY]")
    print(f" iterations = {eval_args.iterations} (random batches), seed = {eval_args.seed}")
    print(f" sample_p_zs_given_zt             mean {nt.mean().item()*1000:.2f} ms  std {nt.std(unbiased=False).item()*1000:.2f} ms")
    print(f" sample_p_zs_given_zt_uncertainty mean {ut.mean().item()*1000:.2f} ms  std {ut.std(unbiased=False).item()*1000:.2f} ms")
    print(f" MC = {int(eval_args.profile_mc_times)}  inner iters = {int(eval_args.profile_n_iters)}\n")


if __name__ == "__main__":
    main()
