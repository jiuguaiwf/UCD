import torch


def build_diagonal_laplace_state(linear_layer, feature_square_sum, prior_precision=1.0, obs_noise=1.0):
    if linear_layer.bias is not None:
        expected_dim = linear_layer.in_features + 1
    else:
        expected_dim = linear_layer.in_features

    if feature_square_sum.numel() != expected_dim:
        raise ValueError(
            f"Expected {expected_dim} feature statistics, got {feature_square_sum.numel()}."
        )

    obs_noise = max(float(obs_noise), 1e-8)
    precision_diag = float(prior_precision) + feature_square_sum / (obs_noise ** 2)
    std_diag = precision_diag.rsqrt()

    state = {
        'weight_mean': linear_layer.weight.detach().clone(),
        'posterior_precision_diag': precision_diag.detach().clone(),
        'posterior_std_diag': std_diag.detach().clone(),
        'prior_precision': torch.tensor(float(prior_precision)),
        'obs_noise': torch.tensor(obs_noise),
    }
    if linear_layer.bias is not None:
        state['bias_mean'] = linear_layer.bias.detach().clone()
    else:
        state['bias_mean'] = None
    return state


def sample_diagonal_last_layer(features, laplace_state, n_samples):
    if laplace_state is None:
        raise RuntimeError("Last-layer Laplace state has not been initialized.")

    device = features.device
    dtype = features.dtype

    weight_mean = laplace_state['weight_mean'].to(device=device, dtype=dtype)
    bias_mean = laplace_state['bias_mean']
    if bias_mean is not None:
        bias_mean = bias_mean.to(device=device, dtype=dtype)
        bias_column = torch.ones(features.size(0), 1, device=device, dtype=dtype)
        design = torch.cat([features, bias_column], dim=1)
        param_mean = torch.cat([weight_mean, bias_mean.unsqueeze(1)], dim=1)
    else:
        design = features
        param_mean = weight_mean

    std_diag = laplace_state['posterior_std_diag'].to(device=device, dtype=dtype)
    noise = torch.randn(
        n_samples,
        param_mean.size(0),
        param_mean.size(1),
        device=device,
        dtype=dtype,
    )
    sampled_params = param_mean.unsqueeze(0) + noise * std_diag.view(1, 1, -1)
    sampled_outputs = torch.einsum('md,sod->smo', design, sampled_params)
    return sampled_outputs
