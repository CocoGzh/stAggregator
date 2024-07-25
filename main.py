import sys

print(sys.path)
sys.path.append('E:\\学习\\6科研\\论文\\博士论文\\scTracer\\code\\')
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scTracer import scTracer, label_transfer
from scTracer import metrics

# config
sc.set_figure_params(dpi=300, figsize=(4, 4), frameon=False)  # TODO 是否画边框
path_results = './log/test_30000/'

# load data
# 自己预处理好，只提供example，不提供脚本
adata1 = sc.read_h5ad('./analysis/2merfish测试/stRNA_sub.h5ad')
adata1.X = csr_matrix(adata1.X)
sc.pp.subsample(adata1, n_obs=3000)
adata2 = sc.read_h5ad('./analysis/2merfish测试/scRNA_processed.h5ad')
sc.pp.subsample(adata2, n_obs=3000)
adata2.X = csr_matrix(adata2.X)
adata1.obsm['spatial'] = adata1.obs.loc[:, ['Centroid_X', 'Centroid_Y']].to_numpy()
adata2.obsm['spatial'] = np.random.rand(len(adata2), 2) * 10000

# preprocess
# 自己预处理好，提供脚本
# sc.pp.normalize_total(adata1)
# sc.pp.log1p(adata1)
# sc.pp.normalize_total(adata2)
# sc.pp.log1p(adata2)
# adata = sc.concat([adata1, adata2])

# data_list = [adata1, adata2]
data_list = [adata2]
batch_key = 'batch'
# batch_names = ['adata1', 'adata2']
batch_names = ['adata2']
from scTracer.data import process_adata

adata = process_adata(data_list,
                      batch_key=batch_key,
                      batch_categories=batch_names,
                      join='inner',
                      min_features=0,
                      min_cells=0,
                      target_sum=None,
                      n_top_features=2000,
                      MinMaxScale=True,
                      chunk_size=20000,
                      )

# cal graph
from scTracer.data import process_graph

# rad_cutoff_list = [0, 0]
rad_cutoff_list = [0]
adata = process_graph(adata, data_list, rad_cutoff_list=rad_cutoff_list)

# train
adata_scCorrect = scTracer(adata=adata, max_iteration=1000, outdir=path_results, assess=True, batch_size=128,
                           plot_umap=True, impute=True)
