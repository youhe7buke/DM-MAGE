import torch
import numpy as np
from GAE import IGAE
from utils import setup_seed, get_adj_multi_metric
from train import Pretrain_gae
import pandas as pd
import scanpy as sc
from preprocess import prepro, normalize
from utils import get_adj
from scipy.sparse import coo_matrix
import h5py
import opt
from opt import args
import os
setup_seed(1)
from torch.utils.data import Dataset, DataLoader
print("use cuda: {}".format(opt.args.cuda))
device = torch.device("cuda" if opt.args.cuda else "cpu")

class LoadDataset(Dataset):

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), \
               torch.from_numpy(np.array(idx))

# dataset_list1=[
#     "PBMC", "klein","kidney","romanov","Human1", "Human2", "Human3","Human4", "Mouse1", "Mouse2", "Zeisel", "HumanLiver"]
# dataset_list2=[
#     "Adam", "Chen","Muraro", "Pollen", "Quake_10x_Limb_Muscle",  "Quake_Smart-seq2_Diaphragm", "Quake_Smart-seq2_Heart", "Quake_Smart-seq2_Limb_Muscle",
#    "Quake_Smart-seq2_Lung","Wang_Lung",]
# dataset_list3=["Yan","Camp_Brain","Camp_Liver","Baron","biase","goolam","Human","Mouse","Xin","Tasic"]

if args.model_save_path is None:
    args.model_save_path = 'model/model_save_gae/{}_gae.pkl'.format(args.name)
opt.args.data_path = 'data/{}.txt'.format(opt.args.name)
opt.args.label_path = 'data/{}_label.txt'.format(opt.args.name)
# opt.args.graph_k_save_path = 'graph/{}{}_graph.txt'.format(opt.args.name, opt.args.k)
opt.args.graph_save_path = 'graph/{}_graph.txt'.format(opt.args.name)

# 创建保存模型的目录
os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)


# data_mat = h5py.File('../dataset/{}.h5'.format(args.name))
# x = np.array(data_mat['X'])
# y = np.array(data_mat['Y'])
# data_mat.close()

x, y = prepro('/data/home/wangchi/scDMMGAE-main/dataset/{}/data.h5'.format(args.name))

# x = np.array(pd.read_csv('../dataset3/{}/count.csv'.format(args.name), header=None))
# y = np.array(pd.read_csv('../dataset3/{}/label.csv'.format(args.name), header=None))

if opt.args.name == "klein" or opt.args.name == "romanov" or opt.args.name == "Baron" or opt.args.name == "biase" or opt.args.name == "goolam" or opt.args.name == "Xin":
    x = x.T
else:
    x = x

x = np.ceil(x).astype(np.float32)
y = y.astype(np.float32)
cluster_number = int(max(y) - min(y) + 1)
print(cluster_number)
args.n_clusters = cluster_number
adata = sc.AnnData(x)
adata = normalize(adata, filter_min_counts=True, highly_genes=2000, size_factors=True,
                  normalize_input=False, logtrans_input=True)
print(adata)

Nsample1, Nfeature = np.shape(adata.X)
y = y.reshape(Nsample1, )

# 解析距离度量列表
distance_metrics = opt.args.distance_metrics.split(',')

# 使用多种距离度量计算邻接矩阵
adj_list, adjn_list = get_adj_multi_metric(adata.X, k=10, pca=50, metrics=distance_metrics)

# 将所有邻接矩阵转换为PyTorch张量
adjn_tensors = []
for adjn in adjn_list:
    adjn_tensor = torch.from_numpy(adjn.astype(np.float32)).to(device)
    adjn_tensors.append(adjn_tensor)

X = torch.from_numpy(adata.X)

dataset = LoadDataset(X)
data = torch.Tensor(dataset.x).to(device)
label = y
args.n_components = Nfeature
model_gae = IGAE(
    gae_n_enc_1=opt.args.gae_n_enc_1,
    gae_n_enc_2=opt.args.gae_n_enc_2,
    gae_n_enc_3=opt.args.gae_n_enc_3,
    gae_n_enc_4=opt.args.gae_n_enc_4,
    gae_n_dec_1=opt.args.gae_n_dec_1,
    gae_n_dec_2=opt.args.gae_n_dec_2,
    gae_n_dec_3=opt.args.gae_n_dec_3,
    gae_n_dec_4=opt.args.gae_n_dec_4,
    n_input=opt.args.n_components,
).to(device)

Pretrain_gae(model_gae, data, adjn_tensors, y, opt.args.gamma_value)

# 在训练完成后添加可视化
if args.visualize:
    from visualization import visualize_tsne
    print("正在生成t-SNE可视化...")
    visualize_tsne(args.model_save_path, args.name)

