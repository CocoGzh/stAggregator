#!/usr/bin/env python

import math
import numpy as np
import dgl
import copy
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.autograd import Function
from .utils import permutation
from dgl.nn.pytorch import GraphConv

activation = {
    'relu': nn.ReLU(),
    'rrelu': nn.RReLU(),
    'sigmoid': nn.Sigmoid(),
    'leaky_relu': nn.LeakyReLU(),
    'tanh': nn.Tanh(),
    '': None
}


class ASRNormBN1d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        '''

        :param dim: C of N,C
        '''
        super(ASRNormBN1d, self).__init__()
        self.eps = eps
        self.num_channels = dim
        self.stan_mid_channel = self.num_channels // 2
        self.rsc_mid_channel = self.num_channels // 16

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.standard_encoder = nn.Linear(dim, self.stan_mid_channel)  # 16
        self.rescale_encoder = nn.Linear(dim, self.rsc_mid_channel)

        # standardization
        self.standard_mean_decoder = nn.Sequential(
            self.relu,
            nn.Linear(self.stan_mid_channel, dim)
        )

        self.standard_std_decoder = nn.Sequential(
            self.relu,
            nn.Linear(self.stan_mid_channel, dim),
            self.relu
        )

        # Rescaling
        self.rescale_beta_decoder = nn.Sequential(
            self.relu,
            nn.Linear(self.rsc_mid_channel, dim),
            self.tanh
        )

        self.rescale_gamma_decoder = nn.Sequential(
            self.relu,
            nn.Linear(self.rsc_mid_channel, dim),
            self.sigmoid
        )

        self.lambda_mu = nn.Parameter(torch.empty(1))
        self.lambda_sigma = nn.Parameter(torch.empty(1))

        self.lambda_beta = nn.Parameter(torch.empty(1))
        self.lambda_gamma = nn.Parameter(torch.empty(1))

        self.bias_beta = nn.Parameter(torch.empty(dim))
        self.bias_gamma = nn.Parameter(torch.empty(dim))

        self.drop_out = nn.Dropout(p=0.3)

        # init lambda and bias
        with torch.no_grad():
            init.constant_(self.lambda_mu, self.sigmoid(torch.tensor(-3)))
            init.constant_(self.lambda_sigma, self.sigmoid(torch.tensor(-3)))
            init.constant_(self.lambda_beta, self.sigmoid(torch.tensor(-5)))
            init.constant_(self.lambda_gamma, self.sigmoid(torch.tensor(-5)))
            init.constant_(self.bias_beta, 0.)
            init.constant_(self.bias_gamma, 1.)

    def forward(self, x):
        '''

        :param x: N,C
        :return:
        '''
        N, C = x.size()
        x_mean = torch.mean(x, dim=0)
        x_std = torch.sqrt(torch.var(x, dim=0)) + self.eps

        # standardization
        x_standard_mean = self.standard_mean_decoder(self.standard_encoder(self.drop_out(x_mean.view(1, -1)))).squeeze()
        x_standard_std = self.standard_std_decoder(self.standard_encoder(self.drop_out(x_std.view(1, -1)))).squeeze()

        mean = self.lambda_mu * x_standard_mean + (1 - self.lambda_mu) * x_mean
        std = self.lambda_sigma * x_standard_std + (1 - self.lambda_sigma) * x_std

        mean = mean.reshape((1, C))
        std = std.reshape((1, C))

        x = (x - mean) / std

        # rescaling
        x_rescaling_beta = self.rescale_beta_decoder(self.rescale_encoder(x_mean.view(1, -1))).squeeze()
        x_rescaling_gamma = self.rescale_gamma_decoder(self.rescale_encoder(x_std.view(1, -1))).squeeze()

        beta = self.lambda_beta * x_rescaling_beta + self.bias_beta
        gamma = self.lambda_gamma * x_rescaling_gamma + self.bias_gamma

        beta = beta.reshape((1, C))
        gamma = gamma.reshape((1, C))

        x = x * gamma + beta

        return x


class DSASRNorm(nn.Module):
    """
    Domain-specific Batch Normalization
    """

    def __init__(self, num_features, n_domain, eps=1e-5, momentum=0.1):
        """
        Parameters
        ----------
        num_features
            dimension of the features
        n_domain
            domain number
        """
        super().__init__()
        self.n_domain = n_domain
        self.num_features = num_features
        self.bns = nn.ModuleList([ASRNormBN1d(num_features) for i in range(n_domain)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, y):
        out = torch.zeros(x.size(0), self.num_features, device=x.device)  # , requires_grad=False)
        for i in range(self.n_domain):
            indices = np.where(y.cpu().numpy() == i)[0]

            if len(indices) > 1:
                out[indices] = self.bns[i](x[indices])
            elif len(indices) == 1:
                out[indices] = x[indices]
        #                 self.bns[i].training = False
        #                 out[indices] = self.bns[i](x[indices])
        #                 self.bns[i].training = True
        return out


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class Encoder(nn.Module):
    """
    Encoder
    """

    def __init__(self, input_dim, h_dim=16):
        """
        Parameters
        ----------
        input_dim
            input dimension
        cfg
            encoder configuration, e.g. enc_cfg = [['fc', 1024, 1, 'relu'],['fc', 10, '', '']]
        """
        super().__init__()
        input_dim = input_dim
        input_dim2 = h_dim * 8
        h_dim = h_dim
        # encode
        self.conv1 = GraphConv(input_dim, input_dim2, weight=True, bias=True, activation=None)
        self.conv2 = GraphConv(input_dim2, input_dim2, weight=True, bias=True, activation=None)

        # reparameterize
        self.conv3 = GraphConv(input_dim2, h_dim, weight=True, bias=True, activation=None)
        self.var_fc = GraphConv(input_dim2, h_dim, weight=True, bias=True, activation=None)

        # DGI
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.disc_o = Discriminator(input_dim2)
        self.disc_h = Discriminator(h_dim)

        # reset_parameters
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.var_fc.reset_parameters()

    def reparameterize(self, mu, var):
        return Normal(mu, var.sqrt()).rsample()

    def forward(self, x, block=None):
        """
        """
        # origin
        o = self.conv1(block, x)
        o = self.conv2(block, o) + o
        h = self.conv3(block, o)
        # origin DGI
        emb_o = self.relu(o)
        g_o = self.sigmoid(F.normalize(emb_o, p=2, dim=1))
        emb_h = self.relu(h)
        g_h = self.sigmoid(F.normalize(emb_h, p=2, dim=1))

        # corrupt
        x_a = permutation(x)
        o_a = self.conv1(block, x_a)
        o_a = self.conv2(block, o_a) + o_a
        h_a = self.conv3(block, o_a)
        # corrupt DGI
        emb_oa = self.relu(o_a)
        g_oa = self.sigmoid(F.normalize(emb_oa, p=2, dim=1))
        emb_ha = self.relu(h_a)
        g_ha = self.sigmoid(F.normalize(emb_ha, p=2, dim=1))

        # DGI
        ret_o = self.disc_o(g_o, emb_o, emb_oa)
        ret_oa = self.disc_o(g_oa, emb_oa, emb_o)

        ret_h = self.disc_h(g_h, emb_h, emb_ha)
        ret_ha = self.disc_h(g_ha, emb_ha, emb_h)

        return h, h, h, [ret_o, ret_h], [ret_oa, ret_ha]


class Decoder(nn.Module):
    """
    Decoder
    """

    def __init__(self, h_dim, input_dim, n_domains=None):
        """
        Parameters
        ----------
        input_dim: input dimension
        h_dim: h_dim
        """
        super().__init__()
        input_dim2 = h_dim * 8
        h_dim = h_dim
        # encode
        self.conv1 = GraphConv(h_dim, input_dim2, weight=True, bias=True, activation=None)
        self.norm = DSASRNorm(input_dim2, n_domains)
        self.act1 = nn.LeakyReLU()

        self.conv2 = GraphConv(input_dim2, input_dim, weight=True, bias=True, activation=None)
        self.act2 = nn.Sigmoid()

        # reset_parameters
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        # self.norm.reset_parameters()
        # self.norm2.reset_parameters()

    def forward(self, x, y=None, block=None):
        o = self.act1(self.norm(self.conv1(block, x), y))
        o = self.conv2(block, o)
        o_sig = self.act2(o)
        return o, o_sig
