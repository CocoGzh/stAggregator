import scanpy as sc
import pandas as pd
import numpy as np
import torch
import dgl
import random
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from stAggregator import stAggregator
from stAggregator import metrics
from stAggregator.data import process_adata, process_graph, mclust_R
from stAggregator.data import process_graph
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
dgl.random.seed(seed)


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
    # adata.obs['label_refined'] = np.array(new_type)

    return new_type


def mclust_R_smooth(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=42):
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

    adata.obs['mclust'] = refine_label(adata, 50, key='mclust')

    return adata
from stAggregator.metrics import evaluate_all
ARI_list = []
NMI_list = []
metrics_list = []
Batch_list = []
adj_list = []
# section_ids = ['151673','151674','151675','151676']
section_ids = [
    '151507.h5ad',
    '151508.h5ad',
    '151509.h5ad',
    '151510.h5ad',

    '151673.h5ad',
    '151674.h5ad',
    '151675.h5ad',
    '151676.h5ad',

    '151669.h5ad',
    '151670.h5ad',
    '151671.h5ad',
    '151672.h5ad']
cluster_num = [7, 7, 7, 7, 7, 7, 7, 7, 5, 5, 5, 5]

for i, section_id in enumerate(section_ids):
    print(section_id)
    adata1 = sc.read_h5ad(f'./analysis/1DLPFC/{section_id}')
    adata1.obs_names_make_unique()
    adata1.var_names_make_unique()
    data_list = [adata1]
    batch_key = 'batch'
    batch_names = ['adata1']

    adata = process_adata(data_list, batch_key=batch_key, batch_categories=batch_names, n_top_features=3000)
    rad_cutoff_list = [150]
    adata = process_graph(adata, data_list, rad_cutoff_list=rad_cutoff_list)
    path_results = f'./log/integration/'
    adata_scCorrect = stAggregator(adata=adata, max_iteration=80, outdir=path_results, h_dim=16,
                                   batch_size=19999, impute=False, early_stop=False, random_seed=42)

    # n_clusters=7
    adata_scTracer = mclust_R_smooth(adata_scCorrect, used_obsm='X_stAggregator', num_cluster=cluster_num[i])
    adata_scTracer.obs['mclust'] = adata_scTracer.obs['mclust']
    # stagate
    sc.settings.set_figure_params(facecolor='white', figsize=(2, 2))
    # sc.pl.embedding(adata_scTracer, basis='spatial', color=['Region', 'mclust', ])

    adata_new = adata_scTracer[~adata_scTracer.obs.loc[:, 'Region'].isnull(), :]
    ari = ari_score(adata_new.obs['Region'], adata_new.obs['mclust'])
    nmi = normalized_mutual_info_score(adata_new.obs['Region'], adata_new.obs['mclust'])
    print('mclust, ARI = %01.3f' % ari)
    print('mclust, NMI = %01.3f' % nmi)
    ARI_list.append(ari)
    NMI_list.append(nmi)
    metrics = evaluate_all(adata_new, gt_key='Region', pred_key='mclust')
    metrics_list.append(metrics)
    # break