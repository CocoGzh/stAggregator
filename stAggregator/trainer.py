#!/usr/bin/env python
import pandas as pd
import torch
import numpy as np
import os
import scanpy as sc

# from .data import load_data, preprocess
from .net.model import Model
from .net.utils import clear_fig, EarlyStopping
from .logger import create_logger
import random
from .data import generate_dataloader

def stAggregator(
        adata=None,
        h_dim=64,
        batch_size=1024,
        lr=1e-3,
        max_iteration=1000,
        random_seed=42,
        gpu=0,
        early_stop=False,
        eval=False,
        impute=None,

        verbose=False,
        outdir='output/',
):
    """

    Parameters
    ----------

    batch_size
        Number of samples per batch to load. Default: 64.
    lr
        Learning rate. Default: 2e-4.
    max_iteration
        Max iterations for training. Training one batch_size samples is one iteration. Default: 30000.
    seed
        Random seed for torch and numpy. Default: 124.
    gpu
        Index of GPU to use if GPU is available. Default: 0.
    outdir
        Output directory. Default: 'output/'.
    impute
        If True, calculate the imputed gene expression and store it at adata.layers['impute']. Default: False.
    verbose
        Verbosity, True or False. Default: False.
    assess
        If True, calculate the entropy_batch_mixing score and silhouette score to evaluate integration results. Default: False.
    
    Returns
    -------
    The output folder contains:
    adata.h5ad
        The AnnData matrice after batch effects removal. The low-dimensional representation of the data is stored at adata.obsm['latent'].
    checkpoint
        model.pt contains the variables of the model and config.pt contains the parameters of the model.
    umap.pdf 
        UMAP plot for visualization.
    """

    seed = random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(gpu)
    else:
        device = 'cpu'

    os.makedirs(outdir + 'checkpoint', exist_ok=True)
    log = create_logger('', fh=outdir + 'log.txt')

    trainloader, testloader, dgl_graph, mode = generate_dataloader(adata, batch_size)
    x_dim, n_domain = adata.shape[1], len(adata.obs['batch'].cat.categories)

    # model config
    model = Model(x_dim, h_dim, n_domain=n_domain)

    # train
    model.fit(
        adata,
        trainloader,
        lr=lr,
        max_iteration=max_iteration,
        device=device,
        early_stop=early_stop,
        outdir=outdir,
        verbose=verbose,
        mode=mode,
        dgl_graph=dgl_graph
    )

    # store
    # adata.obsm['X_generate'], adata.obsm['batch_id'] = model.encodeBatch(testloader, device=device, eval=eval,
    #                                                                      out='generate')
    adata.obsm['X_stAggregator'] = model.encodeBatch(testloader, out='latent', device=device, eval=eval, size=len(adata))  # save latent rep
    adata.obsm['X_emb'] = adata.obsm['X_stAggregator']
    if impute:
        adata.layers['impute'] = model.encodeBatch(testloader, out='impute', device=device, eval=eval, size=len(adata))
    log.info('Output dir: {}'.format(outdir))

    model.to('cpu')
    del model
    del adata.uns['adj']  # TODO can not store scipy sparse matrix as HDF5
    adata.write(outdir + 'adata_stAggregator.h5ad', compression='gzip')

    return adata
