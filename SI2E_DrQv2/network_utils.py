import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

def flatten_two_dims(x):
    return x.view(-1, *x.size()[2:])

def unflatten_first_dim(x, sh):
    return x.view(sh[0], sh[1], *x.size()[1:])

def normal_parse_params(params, min_sigma=0.0):
    n = params.size(0)
    d = params.size(-1)                    # channel
    mu = params[..., :d // 2]              # 前一半是均值
    sigma_params = params[..., d // 2:]    # 后一半是标准差的参数
    sigma = torch.nn.functional.softplus(sigma_params)
    sigma = torch.clamp(sigma, min=min_sigma, max=1e5)  # 限制标准差的范围

    distr = td.Normal(loc=mu, scale=sigma)   # 创建正态分布
    return distr

class ResBlock(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super(ResBlock, self).__init__()
        self.dense1 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.dense2 = nn.Linear(hidden_size + action_dim, hidden_size)

    def forward(self, x, a):
        res = F.leaky_relu(self.dense1(torch.cat([x, a], dim=-1)))
        res = self.dense2(torch.cat([res, a], dim=-1))
        return x + res

class TransitionNetwork(nn.Module):
    def __init__(self, rep_dim, action_dim, hidden_size=256):
        super(TransitionNetwork, self).__init__()
        self.dense1 = nn.Linear(rep_dim + action_dim, hidden_size)
        self.residual_block1 = ResBlock(action_dim, hidden_size)
        self.residual_block2 = ResBlock(action_dim, hidden_size)
        self.dense2 = nn.Linear(hidden_size + action_dim, hidden_size)

    def forward(self, x, a):
        x = F.leaky_relu(self.dense1(torch.cat([x, a], dim=-1)))  # (batch_size * seq_len, 256)
        x = self.residual_block1(x, a)                           # (batch_size * seq_len, 256)
        x = self.residual_block2(x, a)                           # (batch_size * seq_len, 256)
        x = self.dense2(torch.cat([x, a], dim=-1))                 # (batch_size * seq_len, 256)
        # x = unflatten_first_dim(x, sh)                             # shape = (batch_size, seq_len, 256)
        return x

class GenerativeNetworkGaussianFix(nn.Module):
    def __init__(self, insize=512, hidsize=256, outsize=512):
        super(GenerativeNetworkGaussianFix, self).__init__()
        self.outsize = outsize
        self.dense1 = nn.Linear(insize, hidsize)
        self.dense2 = nn.Linear(hidsize, outsize)
        self.var_single = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
        self.residual_block1 = nn.Sequential(
            nn.Linear(hidsize, hidsize),
            nn.LeakyReLU(),
            nn.Linear(hidsize, hidsize)
        )
        self.residual_block2 = nn.Sequential(
            nn.Linear(hidsize, hidsize),
            nn.LeakyReLU(),
            nn.Linear(hidsize, hidsize)
        )
        self.residual_block3 = nn.Sequential(
            nn.Linear(outsize, outsize),
            nn.LeakyReLU(),
            nn.Linear(outsize, outsize)
        )

    def forward(self, z):
        sh = z.shape
        
        x = F.leaky_relu(self.dense1(z))
        x = x + self.residual_block1(x)
        x = x + self.residual_block2(x)
        
        # Variance
        var_tile = self.var_single.expand(sh[0], self.outsize)

        # Mean
        x = F.leaky_relu(self.dense2(x))
        x = x + self.residual_block3(x)

        x = torch.cat([x, var_tile], dim=-1)
        return x

class GenerativeNetworkGaussian(nn.Module):
    def __init__(self, hidsize=256, outsize=512):
        super(GenerativeNetworkGaussian, self).__init__()
        self.dense1 = nn.Linear(128, hidsize)
        self.dense2 = nn.Linear(hidsize, outsize)
        self.dense3 = nn.Linear(outsize, outsize * 2)

        self.residual_block1 = nn.Sequential(
            nn.Linear(hidsize, hidsize),
            nn.LeakyReLU(),
            nn.Linear(hidsize, hidsize)
        )
        self.residual_block2 = nn.Sequential(
            nn.Linear(hidsize, hidsize),
            nn.LeakyReLU(),
            nn.Linear(hidsize, hidsize)
        )
        self.residual_block3 = nn.Sequential(
            nn.Linear(outsize, outsize),
            nn.LeakyReLU(),
            nn.Linear(outsize, outsize)
        )

    def forward(self, z):
        sh = z.shape
        z = flatten_two_dims(z)

        x = F.leaky_relu(self.dense1(z))
        x = x + self.residual_block1(x)
        x = x + self.residual_block2(x)
        x = F.leaky_relu(self.dense2(x))
        x = x + self.residual_block3(x)
        x = self.dense3(x)
        x = unflatten_first_dim(x, sh)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, repr_dim, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.output_dim = output_dim
        self.dense1 = nn.Linear(repr_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(output_dim)

    def forward(self, x, ln=False):
        x = self.dense1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dense2(x)
        x = self.ln2(x)
        return x

class ContrastiveHead(nn.Module):
    def __init__(self, input_dim):
        super(ContrastiveHead, self).__init__()
        self.W = nn.Parameter(torch.rand(input_dim, input_dim))

    def forward(self, z_a, z_pos):
        Wz = torch.matmul(self.W, z_pos.t())
        logits = torch.matmul(z_a, Wz)
        max_logits = torch.max(logits, dim=1, keepdim=True)[0]
        logits = logits - max_logits
        return logits
