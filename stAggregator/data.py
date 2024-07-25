#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import issparse, csr

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

from anndata import AnnData
import scanpy as sc
from sklearn.preprocessing import maxabs_scale, MaxAbsScaler

from glob import glob

np.warnings.filterwarnings('ignore')
CHUNK_SIZE = 20000


def batch_scale(adata, chunk_size=CHUNK_SIZE):
    """
    Batch-specific scale data
    
    Parameters
    ----------
    adata
        AnnData
    chunk_size
        chunk large data into small chunks
    
    Return
    ------
    AnnData
    """
    for b in adata.obs['batch'].unique():
        idx = np.where(adata.obs['batch'] == b)[0]
        scaler = MaxAbsScaler(copy=False).fit(adata.X[idx])
        for i in range(int(np.ceil(len(idx) / chunk_size))):
            adata.X[idx[i * chunk_size:(i + 1) * chunk_size]] = scaler.transform(
                adata.X[idx[i * chunk_size:(i + 1) * chunk_size]])

    return adata


def process_adata(data_list,
                  batch_key='batch',
                  batch_categories=None,
                  join='inner',
                  n_top_features=2000,
                  chunk_size=CHUNK_SIZE,
                  ):
    for i, temp in enumerate(data_list):
        if not isinstance(temp.X, csr.csr_matrix):
            data_list[i].X = scipy.sparse.csr_matrix(temp.X)
        if 'spatial' not in temp.obsm.keys():
            data_list[i].obsm['spatial'] = np.random.rand(len(data_list[i]), 2)
            print(f'spatial not in {i}th adata.obsm')

    adata = sc.concat([*data_list], join=join, label=batch_key, keys=batch_categories)
    adata.obs['batch'] = adata.obs['batch'].astype('category')
    if not isinstance(adata.X, csr.csr_matrix):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    # adata = adata[:, [gene for gene in adata.var_names if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]
    # counts
    adata.layers['counts'] = adata.X
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key=batch_key, flavor='seurat_v3')
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # sc.pp.scale(adata, zero_center=False, max_value=10)
    adata.raw = adata
    # hvg
    # sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key=batch_key)
    adata = adata[:, adata.var.highly_variable]
    # scale
    adata_scale = batch_scale(adata.copy(), chunk_size=chunk_size)
    adata.obsm['feature_scale'] = adata_scale.X
    return adata


def cal_spatial_net(adata, rad_cutoff=None, k_cutoff=None, max_neigh=100, model='Radius', verbose=True):
    """\
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less
        than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """
    import sklearn.neighbors
    import scipy.sparse as sp
    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm['spatial'])
    coor.index = adata.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]

    KNN_list = []
    # for it in range(indices.shape[0]):
    #     KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_list = [pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])) for it in
                range(indices.shape[0])]
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    # self_loops = pd.DataFrame(zip(Spatial_Net['Cell1'].unique(), Spatial_Net['Cell1'].unique(),
    #                  [0] * len((Spatial_Net['Cell1'].unique())))) ###add self loops
    # self_loops.columns = ['Cell1', 'Cell2', 'Distance']
    # Spatial_Net = pd.concat([Spatial_Net, self_loops], axis=0)

    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata.n_obs))
    adata.uns['Spatial_Net'] = Spatial_Net

    #########
    X = pd.DataFrame(adata.X.toarray()[:, ], index=adata.obs.index, columns=adata.var.index)
    cells = np.array(X.index)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    Spatial_Net = adata.uns['Spatial_Net']
    G_df = Spatial_Net.copy()
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])  # TODO self-loop
    adata.uns['adj'] = G
    return adata


def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    import ot
    import scipy.sparse as sp
    position = adata.obsm['spatial']

    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]

    adata.obsm['distance_matrix'] = distance_matrix

    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot])
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1

    adata.obsm['graph_neigh'] = interaction

    # # transform adj to symmetrical adj
    # adj = interaction
    # adj = adj + adj.T
    # adj = np.where(adj > 1, 1, adj)
    # adata.obsm['adj'] = adj

    G = sp.coo_matrix(adata.obsm['graph_neigh'])
    G = G + sp.eye(G.shape[0])  # TODO self-loop
    adata.uns['adj'] = G
    return adata


def process_graph(adata, data_list, rad_cutoff_list=None, n_neighbors=3, max_neigh=100):
    data_list = [cal_spatial_net(adata_temp, rad_cutoff=rad_cutoff_list[i], max_neigh=max_neigh)
                 for i, adata_temp in enumerate(data_list)]
    # data_list = [construct_interaction(adata_temp, n_neighbors=n_neighbors) for i, adata_temp in enumerate(data_list)]
    adj_list = [item.uns['adj'] for item in data_list]

    from scipy.sparse import block_diag
    adj_concat = block_diag(adj_list)
    adata.uns['adj'] = adj_concat
    return adata


def my_collate(batch):
    r"""Custom collate function for dealing with custom types."""

    # Return both collated tensors and custom types
    return None, None, batch[0]


class FullBatchDataset(Dataset):
    def __init__(self, graph):
        self.graph = graph

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.graph


def generate_dataloader(adata, batch_size=64):
    import dgl
    from scipy.sparse import coo_matrix
    coo = coo_matrix(adata.uns['adj'])
    row = torch.from_numpy(coo.row).to(torch.long)
    col = torch.from_numpy(coo.col).to(torch.long)
    dgl_graph = dgl.graph(tuple([row, col]))
    # feature
    dgl_graph.ndata['feature'] = torch.tensor(adata.X.todense(), dtype=torch.float32)
    dgl_graph.ndata['feature_scale'] = torch.tensor(adata.obsm['feature_scale'].todense(), dtype=torch.float32)
    dgl_graph.ndata['batch'] = torch.tensor(adata.obs.loc[:, 'batch'].cat.codes.to_numpy(), dtype=torch.int64)
    dgl_graph.ndata['index'] = torch.tensor(np.arange(0, len(adata)), dtype=torch.int64)
    # dgi label
    one_matrix = np.ones([adata.n_obs, 1])
    zero_matrix = np.zeros([adata.n_obs, 1])
    DGI_label = np.concatenate([one_matrix, zero_matrix], axis=1)
    dgl_graph.ndata['DGI_label'] = torch.tensor(DGI_label, dtype=torch.float32)

    if batch_size < len(adata):
        # mini batch
        # dataset & dataloader
        # sampler
        sampler_list = [5]  # TODO 超参数
        # sampler_list = [5] * (1 + self.params['gnn_layer_num'] + 1)     # 每层只取5个邻居
        sampler = dgl.dataloading.ShaDowKHopSampler(sampler_list)
        # train_nids是全部的，loss时做选择
        train_nids = dgl_graph.ndata['index']
        # dataloader
        trainloader = dgl.dataloading.NodeDataLoader(dgl_graph,
                                                     train_nids,
                                                     sampler,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     drop_last=False)
        testloader = dgl.dataloading.NodeDataLoader(dgl_graph,
                                                    train_nids,
                                                    sampler,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    drop_last=False)
        mode = 'mini_batch'
    elif batch_size >= len(adata):
        # full batch
        st_dataset = FullBatchDataset(dgl_graph)
        trainloader = DataLoader(
            st_dataset,
            batch_size=1,
            collate_fn=my_collate,
            drop_last=False,
            shuffle=False,
            num_workers=0
        )
        testloader = DataLoader(
            st_dataset,
            batch_size=1,
            collate_fn=my_collate,
            drop_last=False,
            shuffle=False,
            num_workers=0
        )
        mode = 'full_batch'
    else:
        raise NotImplementedError
    return trainloader, testloader, dgl_graph, mode


def refine_label(adata, radius=50, key='label'):
    import ot
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]

    return new_type


def mclust_R_smooth(adata, num_cluster, modelNames='EEE', used_obsm='X_pca', radius=50, random_seed=42):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """

    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    adata.obsm['mclust_prob'] = np.array(res[-3])

    adata.obs['mclust'] = refine_label(adata, radius=radius, key='mclust')

    return adata
