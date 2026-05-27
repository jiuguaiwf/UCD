import torch
import torch.nn as nn
from egnn.egnn_new import EGNN as BaseEGNN, GNN
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
import numpy as np

try:
    from egnn.egnn_new_dropout import EGNN as DropoutEGNN
except ModuleNotFoundError:
    DropoutEGNN = None

def compute_mc_dropout_scalar_uncertainty(mc_outputs: torch.Tensor, reduce: str = 'mean'):
    # mc_outputs: [K, ...]
    var = mc_outputs.var(dim=0, unbiased=False)
    if reduce == 'mean':
        return var.mean()
    elif reduce == 'max':
        return var.max()
    else:
        raise ValueError("reduce must be 'mean' or 'max'")


class EGNN_dynamics_QM9(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                 condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                 inv_sublayers=2, sin_embedding=False, normalization_factor=100,
                 aggregation_method='sum', egnn_variant='base'):
        super().__init__()
        self.mode = mode
        self.egnn_variant = egnn_variant
        self.uncertainty_method = 'laplace' if egnn_variant == 'base' else 'dropout'
        if mode == 'egnn_dynamics':
            egnn_cls = BaseEGNN
            if egnn_variant == 'dropout':
                if DropoutEGNN is None:
                    raise ImportError(
                        "Dropout EGNN requires optional dependencies such as torch_geometric."
                    )
                egnn_cls = DropoutEGNN
            self.egnn = egnn_cls(
                in_node_nf=in_node_nf + context_node_nf, in_edge_nf=1,
                hidden_nf=hidden_nf, device=device, act_fn=act_fn,
                n_layers=n_layers, attention=attention, tanh=tanh, norm_constant=norm_constant,
                inv_sublayers=inv_sublayers, sin_embedding=sin_embedding,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method)
            self.in_node_nf = in_node_nf
        elif mode == 'gnn_dynamics':
            self.gnn = GNN(
                in_node_nf=in_node_nf + context_node_nf + 3, in_edge_nf=0,
                hidden_nf=hidden_nf, out_node_nf=3 + in_node_nf, device=device,
                act_fn=act_fn, n_layers=n_layers, attention=attention,
                normalization_factor=normalization_factor, aggregation_method=aggregation_method)

        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _prepare_inputs(self, t, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]
        node_mask_flat = node_mask.view(bs * n_nodes, 1)
        edge_mask_flat = edge_mask.view(bs * n_nodes * n_nodes, 1)
        xh_flat = xh.view(bs * n_nodes, -1).clone() * node_mask_flat
        x = xh_flat[:, 0:self.n_dims].clone()

        if h_dims == 0:
            h = torch.ones(bs * n_nodes, 1, device=self.device)
        else:
            h = xh_flat[:, self.n_dims:].clone()

        if self.condition_time:
            if np.prod(t.size()) == 1:
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                h_time = t.view(bs, 1).repeat(1, n_nodes).view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)

        if context is not None:
            context_flat = context.view(bs * n_nodes, self.context_node_nf)
            h = torch.cat([h, context_flat], dim=1)

        return {
            'bs': bs,
            'n_nodes': n_nodes,
            'h_dims': h_dims,
            'edges': edges,
            'node_mask_flat': node_mask_flat,
            'edge_mask_flat': edge_mask_flat,
            'x': x,
            'h': h,
        }

    def _forward(self, t, xh, node_mask, edge_mask, context):
        prepared = self._prepare_inputs(t, xh, node_mask, edge_mask, context)
        bs = prepared['bs']
        n_nodes = prepared['n_nodes']
        h_dims = prepared['h_dims']
        edges = prepared['edges']
        node_mask = prepared['node_mask_flat']
        edge_mask = prepared['edge_mask_flat']
        x = prepared['x']
        h = prepared['h']

        if self.mode == 'egnn_dynamics':
            h_final, x_final = self.egnn(h, x, edges, node_mask=node_mask, edge_mask=edge_mask)
            vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        elif self.mode == 'gnn_dynamics':
            xh = torch.cat([x, h], dim=1)
            output = self.gnn(xh, edges, node_mask=node_mask)
            vel = output[:, 0:3] * node_mask
            h_final = output[:, 3:]

        else:
            raise Exception("Wrong mode %s" % self.mode)

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2)

    def set_uncertainty_method(self, method):
        self.uncertainty_method = method

    def set_last_layer_laplace(self, laplace_state):
        if not hasattr(self, 'egnn') or not hasattr(self.egnn, 'set_last_layer_laplace'):
            raise RuntimeError("This EGNN variant does not support last-layer Laplace.")
        self.egnn.set_last_layer_laplace(laplace_state)

    @torch.no_grad()
    def get_last_layer_features(self, t, xh, node_mask, edge_mask, context=None):
        if self.mode != 'egnn_dynamics' or not hasattr(self.egnn, 'get_last_layer_features'):
            raise RuntimeError("Last-layer features are only available for EGNN dynamics.")
        prepared = self._prepare_inputs(t, xh, node_mask, edge_mask, context)
        features = self.egnn.get_last_layer_features(
            prepared['h'],
            prepared['x'],
            prepared['edges'],
            node_mask=prepared['node_mask_flat'],
            edge_mask=prepared['edge_mask_flat'],
        )
        return features, prepared['node_mask_flat']

    @torch.no_grad()
    def uncertainty(self, t, xh, node_mask, edge_mask, context=None,
                    variance_cal_times: int = 20, reduce: str = 'mean', method=None):
        """
        Must return (net_out, net_out_uncertainty) because en_diffusion.py expects:
            net_out, net_out_uncertainty = self.dynamics.uncertainty(...)
        DiT-style: MC forward passes -> scalar variance.
        """

        # --- If your EGNN class already has its own .uncertainty(), use it ---
        if hasattr(self, "egnn") and hasattr(self.egnn, "uncertainty"):
            prepared = self._prepare_inputs(t, xh, node_mask, edge_mask, context)
            uncertainty_kwargs = {
                'node_mask': prepared['node_mask_flat'],
                'edge_mask': prepared['edge_mask_flat'],
                'N': int(variance_cal_times),
            }
            if self.egnn_variant == 'base':
                uncertainty_kwargs['method'] = method or self.uncertainty_method
            h_mean, unc_scalar = self.egnn.uncertainty(
                prepared['h'], prepared['x'], prepared['edges'],
                **uncertainty_kwargs,
            )
            net_out = self._forward(t, xh, node_mask, edge_mask, context)
            return net_out, unc_scalar

        # --- Fallback: do MC on the dynamics output directly (always works) ---
        mc_out = []
        for _ in range(int(variance_cal_times)):
            mc_out.append(self._forward(t, xh, node_mask, edge_mask, context))
        mc_out = torch.stack(mc_out, dim=0)  # [K, bs, n_nodes, out_dim]

        mean_out = mc_out.mean(dim=0)
        unc_scalar = compute_mc_dropout_scalar_uncertainty(mc_out, reduce=reduce)
        return mean_out, unc_scalar


    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)
