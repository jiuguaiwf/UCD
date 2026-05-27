# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import utils
import argparse
from qm9 import dataset
from qm9.models import get_model
import os
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked
import torch
import time
import pickle
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9.sampling import uncertainty_sample
from qm9.sampling import sample
from qm9.analyze import analyze_stability_for_molecules, analyze_node_distribution
from qm9.path_utils import resolve_qm9_datadir
from qm9.utils import prepare_context, compute_mean_mad
from qm9 import visualizer as qm9_visualizer
import qm9.losses as losses
from egnn.last_layer_laplace import build_diagonal_laplace_state

try:
    from qm9 import rdkit_functions
except ModuleNotFoundError:
    print('Not importing rdkit functions.')


def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)


def get_context(args, data, device, dtype, node_mask, property_norms=None):
    if len(args.conditioning) == 0:
        return None

    context = prepare_context(args.conditioning, data, property_norms).to(device, dtype)
    assert_correctly_masked(context, node_mask)
    return context


def move_laplace_state_to_model(generative_model, laplace_state):
    generative_model.dynamics.set_last_layer_laplace(laplace_state)


@torch.no_grad()
def fit_last_layer_laplace(
        args, generative_model, train_loader, device, dtype,
        property_norms=None, max_batches=200, prior_precision=1.0,
        obs_noise=0.0, cache_path=''):
    if getattr(generative_model.dynamics, 'egnn_variant', 'base') != 'base':
        raise RuntimeError('Last-layer Laplace is only implemented for the base EGNN head.')

    linear_layer = generative_model.dynamics.egnn.embedding_out
    feature_square_sum = None
    residual_square_sum = 0.0
    residual_count = 0
    processed_batches = 0

    generative_model.eval()

    for batch_idx, data in enumerate(train_loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

        batch_size, n_nodes, _ = x.size()

        x = remove_mean_with_mask(x, node_mask)
        check_mask_correct([x, one_hot], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}
        context = get_context(args, data, device, dtype, node_mask, property_norms)
        edge_mask = edge_mask.view(batch_size, n_nodes * n_nodes)

        x, h, _ = generative_model.normalize(x, h, node_mask)

        t_int = torch.randint(
            0, generative_model.T + 1, size=(batch_size, 1), device=device
        ).float()
        t = t_int / generative_model.T
        gamma_t = generative_model.inflate_batch_array(generative_model.gamma(t), x)
        alpha_t = generative_model.alpha(gamma_t, x)
        sigma_t = generative_model.sigma(gamma_t, x)

        eps = generative_model.sample_combined_position_feature_noise(
            n_samples=batch_size, n_nodes=n_nodes, node_mask=node_mask
        )
        xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        z_t = alpha_t * xh + sigma_t * eps

        features, flat_node_mask = generative_model.dynamics.get_last_layer_features(
            t.squeeze(-1), z_t, node_mask, edge_mask, context
        )
        active = flat_node_mask.squeeze(-1) > 0.5
        features = features[active]
        if features.numel() == 0:
            continue

        batch_feature_stats = features.pow(2).sum(dim=0)
        if linear_layer.bias is not None:
            batch_feature_stats = torch.cat(
                [batch_feature_stats, batch_feature_stats.new_tensor([features.size(0)])]
            )
        if feature_square_sum is None:
            feature_square_sum = batch_feature_stats
        else:
            feature_square_sum = feature_square_sum + batch_feature_stats

        net_out = generative_model.phi(z_t, t, node_mask, edge_mask, context)
        residual = ((eps - net_out) * node_mask).pow(2)
        residual_square_sum += residual.sum().item()
        residual_count += int(node_mask.sum().item()) * net_out.size(-1)
        processed_batches += 1

    if processed_batches == 0 or feature_square_sum is None:
        raise RuntimeError('Laplace fitting did not process any training batches.')

    if obs_noise <= 0:
        obs_noise = max((residual_square_sum / max(residual_count, 1)) ** 0.5, 1e-4)

    laplace_state = build_diagonal_laplace_state(
        linear_layer=linear_layer,
        feature_square_sum=feature_square_sum.detach().cpu(),
        prior_precision=prior_precision,
        obs_noise=obs_noise,
    )
    move_laplace_state_to_model(generative_model, laplace_state)

    if cache_path:
        cache_state = {}
        for key, value in laplace_state.items():
            cache_state[key] = value.cpu() if torch.is_tensor(value) else value
        cache_state['processed_batches'] = processed_batches
        torch.save(cache_state, cache_path)

    print(
        f'Fitted last-layer Laplace with {processed_batches} train batches, '
        f'obs_noise={float(obs_noise):.6f}, prior_precision={prior_precision:.4f}'
    )
    return laplace_state


def maybe_load_last_layer_laplace(generative_model, cache_path, device):
    if cache_path == '' or not os.path.exists(cache_path):
        return False

    laplace_state = torch.load(cache_path, map_location=device)
    move_laplace_state_to_model(generative_model, laplace_state)
    print(f'Loaded cached last-layer Laplace state from {cache_path}')
    return True


def analyze_and_save(args, eval_args, device, generative_model,
                     nodes_dist, prop_dist, dataset_info, n_samples=10,
                     batch_size=10, save_to_xyz=False, variance_cal_times=20, dynamic_weights=False, u_max=1):
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    start_time = time.time()
    uncertainty_test = 0
    mean_max_var = 0
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask, traj_uncertainty = uncertainty_sample(
            args, device, generative_model, dataset_info, prop_dist=prop_dist, nodesxsample=nodesxsample, variance_cal_times=variance_cal_times, dynamic_weights=dynamic_weights, u_max=u_max)
        uncertainty_test += sum(traj_uncertainty) / max(len(traj_uncertainty), 1) / (n_samples/batch_size)
        mean_max_var += max(traj_uncertainty) / (n_samples/batch_size)


        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

        current_num_samples = (i+1) * batch_size
        secs_per_sample = (time.time() - start_time) / current_num_samples
        print('\t %d/%d Molecules generated at %.2f secs/sample' % (
            current_num_samples, n_samples, secs_per_sample))

        if save_to_xyz:
            id_from = i * batch_size
            qm9_visualizer.save_xyz_file(
                join(eval_args.model_path, 'eval/analyzed_molecules/'),
                one_hot, charges, x, dataset_info, id_from, name='molecule',
                node_mask=node_mask)

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    stability_dict, rdkit_metrics = analyze_stability_for_molecules(
        molecules, dataset_info)

    return stability_dict, rdkit_metrics, {'mean_var': uncertainty_test, 'mean_max_var': mean_max_var}


def test(args, flow_dp, nodes_dist, device, dtype, loader, partition='Test',
         num_passes=1, property_norms=None):
    flow_dp.eval()
    nll_epoch = 0
    n_samples = 0
    for pass_number in range(num_passes):
        with torch.no_grad():
            for i, data in enumerate(loader):
                # Get data
                x = data['positions'].to(device, dtype)
                node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
                edge_mask = data['edge_mask'].to(device, dtype)
                one_hot = data['one_hot'].to(device, dtype)
                charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

                batch_size = x.size(0)

                x = remove_mean_with_mask(x, node_mask)
                check_mask_correct([x, one_hot], node_mask)
                assert_mean_zero_with_mask(x, node_mask)

                h = {'categorical': one_hot, 'integer': charges}

                context = get_context(args, data, device, dtype, node_mask, property_norms)

                # transform batch through flow
                nll, _, _ = losses.compute_loss_and_nll(args, flow_dp, nodes_dist, x, h, node_mask,
                                                        edge_mask, context)
                # standard nll from forward KL

                nll_epoch += nll.item() * batch_size
                n_samples += batch_size
                if i % args.n_report_steps == 0:
                    print(f"\r {partition} NLL \t, iter: {i}/{len(loader)}, "
                          f"NLL: {nll_epoch/n_samples:.2f}")

    return nll_epoch/n_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/edm_qm9",
                        help='Specify model path')
    parser.add_argument('--n_samples', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--batch_size_gen', type=int, default=100,
                        help='Specify model path')
    parser.add_argument('--save_to_xyz', type=eval, default=False,
                        help='Should save samples to xyz files.')

    parser.add_argument('--variance_cal_times', type=int, default=20,
                        help='Number of posterior samples used for uncertainty estimation')
    parser.add_argument('--dynamic_weights', type=int, default=0,
                        help='Whether enable uncertainty weights in Langevin Dynamics')
    parser.add_argument('--u_max', type=float, default=0.00037,
                        help='Threshold for uncertainty')
    parser.add_argument('--uncertainty_method', type=str, default='laplace',
                        help='dropout | laplace')
    parser.add_argument('--egnn_variant', type=str, default='base',
                        help='base | dropout')
    parser.add_argument('--laplace_n_batches', type=int, default=200,
                        help='Number of train batches to fit the last-layer Laplace posterior. <=0 uses all batches.')
    parser.add_argument('--laplace_prior_precision', type=float, default=1.0,
                        help='Gaussian prior precision for the last-layer Laplace posterior.')
    parser.add_argument('--laplace_obs_noise', type=float, default=0.0,
                        help='Observation noise for Laplace. <=0 estimates it from train residuals.')
    parser.add_argument('--laplace_cache', type=str, default='',
                        help='Optional path to cache the fitted last-layer Laplace posterior.')
    parser.add_argument('--laplace_refit', type=eval, default=False,
                        help='If True, ignore cached Laplace posterior and refit it.')
    parser.add_argument('--datadir', type=str, default=None,
                        help='Override the QM9 data directory. It should contain qm9/train.npz etc.')
    parser.add_argument('--compute_nll', type=eval, default=False,
                        help='If True, also run validation/test NLL after sampling.')

    eval_args, unparsed_args = parser.parse_known_args()

    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'
    args.egnn_variant = eval_args.egnn_variant
    args.datadir = resolve_qm9_datadir(
        eval_args.datadir or getattr(args, 'datadir', None),
        require_processed=True,
    )
    if eval_args.uncertainty_method == 'laplace':
        args.egnn_variant = 'base'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    utils.create_folders(args)
    print(args)

    # Retrieve QM9 dataloaders
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    property_norms = None

    # Load model
    generative_model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)
    generative_model.to(device)

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    generative_model.load_state_dict(flow_state_dict)
    if hasattr(generative_model.dynamics, 'set_uncertainty_method'):
        generative_model.dynamics.set_uncertainty_method(eval_args.uncertainty_method)

    if eval_args.uncertainty_method == 'laplace':
        laplace_cache = eval_args.laplace_cache
        if laplace_cache == '':
            laplace_cache = join(eval_args.model_path, 'last_layer_laplace.pt')

        loaded = False
        if not eval_args.laplace_refit:
            loaded = maybe_load_last_layer_laplace(generative_model, laplace_cache, device)
        if not loaded:
            fit_last_layer_laplace(
                args=args,
                generative_model=generative_model,
                train_loader=dataloaders['train'],
                device=device,
                dtype=dtype,
                property_norms=property_norms,
                max_batches=eval_args.laplace_n_batches,
                prior_precision=eval_args.laplace_prior_precision,
                obs_noise=eval_args.laplace_obs_noise,
                cache_path=laplace_cache,
            )

    # Analyze stability, validity, uniqueness and novelty
    stability_dict, rdkit_metrics, unc_stats = analyze_and_save(
        args, eval_args, device, generative_model, nodes_dist,
        prop_dist, dataset_info, n_samples=eval_args.n_samples,
        batch_size=eval_args.batch_size_gen, save_to_xyz=eval_args.save_to_xyz,
            variance_cal_times=eval_args.variance_cal_times,
            dynamic_weights=eval_args.dynamic_weights,
            u_max=eval_args.u_max)
    print(stability_dict)
    print(unc_stats)

    if rdkit_metrics is not None:
        rdkit_metrics = rdkit_metrics[0]
        print("Validity %.4f, Uniqueness: %.4f, Novelty: %.4f" % (rdkit_metrics[0], rdkit_metrics[1], rdkit_metrics[2]))
    else:
        print("Install rdkit roolkit to obtain Validity, Uniqueness, Novelty")

    log_path = join(eval_args.model_path, 'eval_log_30.txt')
    if eval_args.compute_nll:
        # In GEOM-Drugs the validation partition is named 'val', not 'valid'.
        if args.dataset == 'geom':
            val_name = 'val'
            num_passes = 1
        else:
            val_name = 'valid'
            num_passes = 5

        val_nll = test(args, generative_model, nodes_dist, device, dtype,
                       dataloaders[val_name],
                       partition='Val', property_norms=property_norms)
        print(f'Final val nll {val_nll}')
        test_nll = test(args, generative_model, nodes_dist, device, dtype,
                        dataloaders['test'],
                        partition='Test', num_passes=num_passes,
                        property_norms=property_norms)
        print(f'Final test nll {test_nll}')

        print(f'Overview: val nll {val_nll} test nll {test_nll}', stability_dict, rdkit_metrics)
        with open(log_path, 'w') as f:
            print(f'Overview: val nll {val_nll} test nll {test_nll}',
                  stability_dict, rdkit_metrics,
                  file=f)
    else:
        print('Skipping val/test NLL evaluation.')
        with open(log_path, 'w') as f:
            print('Overview:', stability_dict, rdkit_metrics, unc_stats, file=f)


if __name__ == "__main__":
    main()
