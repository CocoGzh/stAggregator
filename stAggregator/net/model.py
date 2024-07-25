#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import sys
from torch.optim import lr_scheduler
from ..net.utils import clear_fig, EarlyStopping
from .layer import *
from .loss import *


class Model(nn.Module):
    """
    stAggregator framework
    """

    def __init__(self, input_dim, h_dim, n_domain=1):
        """
        Parameters
        ----------
        enc
            Encoder structure config
        dec
            Decoder structure config
        n_domain
            The number of different domains
        """
        super().__init__()

        self.encoder = Encoder(input_dim, h_dim)
        self.decoder = Decoder(h_dim, input_dim, n_domain)
        self.n_domain = n_domain
        self.awl = AutomaticWeightedLoss(2)
        self.x_dim = input_dim
        self.z_dim = h_dim

    def load_model(self, path):
        """
        Load trained model parameters dictionary.
        Parameters
        ----------
        path
            file path that stores the model parameters
        """
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def encodeBatch(
            self,
            dataloader,
            device='cuda',
            out='latent',
            size=None,
            batch_id=None,
            return_idx=False,
            eval=False
    ):
        """
        Inference

        Parameters
        ----------
        dataloader
            An iterable over the given dataset for inference.
        device
            'cuda' or 'cpu' for . Default: 'cuda'.
        out
            The inference layer for output. If 'latent', output latent feature z. If 'impute', output imputed gene expression matrix. Default: 'latent'.
        batch_id
            If None, use batch 0 decoder to infer for all samples. Else, use the corresponding decoder according to the sample batch id to infer for each sample.
        return_idx
            Whether return the dataloader sample index. Default: False.
        eval
            If True, set the model to evaluation mode. If False, set the model to train mode. Default: False.

        Returns
        -------
        Inference layer and sample index (if return_idx=True).
        """
        self.to(device)
        if eval:
            self.eval()
            print('eval mode')
        else:
            self.train()

        indices = np.zeros(size)
        if out == 'latent':
            output = np.zeros((size, self.z_dim))
            for i, (input_nodes, output_nodes, block) in enumerate(dataloader):
                block = block.to(device)
                x, x_scale, y, idx = block.ndata['feature'], block.ndata['feature_scale'], block.ndata['batch'], block.ndata['index']
                z = self.encoder(x, block)[1]  # z, mu, var
                output[idx.detach().cpu().numpy()] = z.detach().cpu().numpy()
                indices[idx.detach().cpu().numpy()] = idx.detach().cpu().numpy()
        elif out == 'impute':
            output = np.zeros((size, self.x_dim))
            for i, (input_nodes, output_nodes, block) in enumerate(dataloader):
                block = block.to(device)
                x, x_scale, y, idx = block.ndata['feature'], block.ndata['feature_scale'], block.ndata['batch'], block.ndata['index']
                z = self.encoder(x, block)[0]  # z, mu, var
                recon_x, recon_x_scale = self.decoder(z, y, block)
                recon_x = recon_x.detach().cpu().numpy()
                recon_x[recon_x < 0] = 0
                output[idx.detach().cpu().numpy()] = recon_x
                # output[idx.detach().cpu().numpy()] = np.where(recon_x > 0, recon_x, 0)
                indices[idx.detach().cpu().numpy()] = idx.detach().cpu().numpy()
        else:
            raise NotImplementedError
        if return_idx:
            return output, indices
        else:
            return output

    def fit(
            self,
            adata,
            dataloader,
            lr=1e-3,
            max_iteration=1000,
            early_stop=False,
            outdir='.',
            device='cuda',
            verbose=False,
            mode='full',
            dgl_graph = None,
    ):
        """
        Fit model

        Parameters
        ----------
        dataloader
            An iterable over the given dataset for training.
        lr
            Learning rate. Default: 2e-4.
        max_iteration
            Max iterations for training. Training one batch_size samples is one iteration. Default: 30000.
        beta
            The co-efficient of KL-divergence when calculate loss. Default: 0.5.
        early_stopping
            EarlyStopping class (definite in utils.py) for stoping the training if loss doesn't improve after a given patience. Default: None.
        device
            'cuda' or 'cpu' for training. Default: 'cuda'.
        verbose
            Verbosity, True or False. Default: False.
        """
        self.to(device)
        if mode == 'full_batch':
            dgl_graph = dgl_graph.to(device)
        optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)
        n_epoch = int(np.ceil(max_iteration / (len(dataloader) + 1)))  # TODO
        # n_epoch = max_iteration  # TODO
        early_stopping = EarlyStopping(patience=int(n_epoch/10), switch=early_stop, checkpoint_file=outdir + '/checkpoint/model.pt')
        with tqdm(range(n_epoch), total=n_epoch, desc='Epochs') as tq:
            for epoch in tq:
                if mode == 'mini_batch':
                    tk0 = tqdm(enumerate(dataloader), total=len(dataloader), leave=False, desc='Iterations',
                               disable=(not verbose))
                    epoch_loss = defaultdict(float)
                    for i, (input_nodes, output_nodes, block) in tk0:
                        block = block.to(device)
                        x, x_scale, y, DGI_label = block.ndata['feature'], block.ndata['feature_scale'], block.ndata['batch'], block.ndata['DGI_label']
                        z, z, z, ret, ret_a = self.encoder(x, block)
                        recon_x, recon_x_scale = self.decoder(z, y, block)
                        # loss
                        recon_loss_mse = F.mse_loss(recon_x, x) * x.size(-1) / 1  ## TO DO
                        sl1_loss = F.binary_cross_entropy_with_logits(ret[0], DGI_label) + F.binary_cross_entropy_with_logits(ret[1], DGI_label)
                        sl2_loss = F.binary_cross_entropy_with_logits(ret_a[0], DGI_label) + F.binary_cross_entropy_with_logits(ret_a[1], DGI_label)
                        loss = {'recon_loss_mse': recon_loss_mse, 'sl1_loss': sl1_loss, 'sl2_loss': sl2_loss}

                        optim.zero_grad()
                        sum(loss.values()).backward()
                        torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=10, norm_type=2)
                        optim.step()

                        for k, v in loss.items():
                            epoch_loss[k] += loss[k].item()

                        info = ','.join(['{}={:.3f}'.format(k, v) for k, v in loss.items()])
                        tk0.set_postfix_str(info)
                else:
                    epoch_loss = defaultdict(float)
                    x, x_scale, y, DGI_label = dgl_graph.ndata['feature'], dgl_graph.ndata['feature_scale'], dgl_graph.ndata['batch'], dgl_graph.ndata['DGI_label']
                    z, z, z, ret, ret_a = self.encoder(x, dgl_graph)
                    recon_x, recon_x_scale = self.decoder(z, y, dgl_graph)
                    recon_loss_mse = F.mse_loss(recon_x, x) * x.size(-1) / 1  ## TO DO
                    # loss
                    sl1_loss = F.binary_cross_entropy_with_logits(ret[0], DGI_label) + F.binary_cross_entropy_with_logits(ret[1], DGI_label)
                    sl2_loss = F.binary_cross_entropy_with_logits(ret_a[0], DGI_label) + F.binary_cross_entropy_with_logits(ret_a[1], DGI_label)
                    loss = {'recon_loss_mse': recon_loss_mse, 'sl1_loss': sl1_loss, 'sl2_loss': sl2_loss}

                    optim.zero_grad()
                    sum(loss.values()).backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=10, norm_type=2)
                    optim.step()

                    for k, v in loss.items():
                        epoch_loss[k] += loss[k].item()

                epoch_info = ', '.join(['{}={:.3f}'.format(k, v) for k, v in epoch_loss.items()])
                epoch_info += f', lr: {optim.param_groups[0]["lr"]}'
                tq.set_postfix_str(epoch_info)

                early_stopping(sum(epoch_loss.values()), self)
                if early_stopping.early_stop:
                    print('EarlyStopping: run {} epoch'.format(epoch + 1))
                    break
        return adata
        
                