import torch
from torch import nn
from torch.nn import Linear, Embedding
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from math import sqrt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def swish(x):
    return x * torch.sigmoid(x)

class ConcreteDropout(nn.Module):
    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()
        self.layer = layer
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        init_min_tensor = torch.tensor(init_min).to(device)
        init_max_tensor = torch.tensor(init_max).to(device)

        self.init_min = torch.log(init_min_tensor) - torch.log(1. - init_min_tensor)
        self.init_max = torch.log(init_max_tensor) - torch.log(1. - init_max_tensor)
        self.p_logit = torch.nn.Parameter(torch.empty(1).uniform_(self.init_min, self.init_max))
        self.p = torch.sigmoid(self.p_logit).to(device)

    def forward(self, x):
        output = self.concrete_dropout(x)
        return self.layer(output)

    def concrete_dropout(self, x):
        eps = 1e-07
        temp = 0.1
        unif_noise = torch.rand_like(x)
        drop_prob = (torch.log(torch.sigmoid(self.p_logit) + eps)
                     - torch.log(1. - torch.sigmoid(self.p_logit) + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1. - unif_noise + eps))
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - torch.sigmoid(self.p_logit)
        x = x * random_tensor / retain_prob
        return x

    def regularize(self):
        weight = torch.flatten(self.layer.weight)
        kr = self.weight_regularizer * torch.sum(weight**2) * (1. - torch.sigmoid(self.p_logit))
        dr = torch.sigmoid(self.p_logit) * torch.log(torch.sigmoid(self.p_logit)) + (1. - torch.sigmoid(self.p_logit)) * torch.log(1. - torch.sigmoid(self.p_logit))
        dr *= self.dropout_regularizer * weight.numel()
        return torch.sum(kr + dr)

class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish, droprate=0.3):
        super(ResidualLayer, self).__init__()
        #self.dropout = ConcreteDropout(Linear(hidden_channels, hidden_channels)).to(device)
        #self.dropout = nn.Dropout(droprate)
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = ConcreteDropout(Linear(hidden_channels, hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.layer.weight, scale=2.0)
        self.lin2.layer.bias.data.fill_(0)

    def forward(self, x):
        reg = self.lin2.regularize()
        return x + self.act(self.lin2(self.act(self.lin1(x)))), reg