import torch
import numpy as np
from scDMMGAE import scDMMGAE
from load_data import LoadDataset, load_graph, construct_graph
import pandas as pd
import scanpy as sc
from preprocess import prepro, normalize
from utils import get_adj
from scipy.sparse import coo_matrix
import h5py
import opt
from opt import args
from torch.utils.data import Dataset, DataLoader
from train import Train, acc_reuslt, nmi_result, f1_result, ari_result
import os
from visualization import visualize_tsne


print("network setting…")


print("use cuda: {}".format(opt.args.cuda))
device = torch.device("cuda" if opt.args.cuda else "cpu")




# dataset_list1=[
#     "PBMC", "klein","kidney","romanov","Human1", "Human2", "Human3","Human4", "Mouse1", "Mouse2", "Zeisel", "HumanLiver"]
# dataset_list2=[
#     "Adam", "Chen","Muraro", "Pollen", "Quake_10x_Limb_Muscle",  "Quake_Smart-seq2_Diaphragm", "Quake_Smart-seq2_Heart", "Quake_Smart-seq2_Limb_Muscle",
#    "Quake_Smart-seq2_Lung","Wang_Lung",]
# dataset_list3=["Yan","Camp_Brain","Camp_Liver","Baron","biase","goolam","Human","Mouse","Xin","Tasic"]



opt.args.data_path = 'data/{}.txt'.format(opt.args.name)
opt.args.label_path = 'data/{}_label.txt'.format(opt.args.name)
opt.args.graph_save_path = 'graph/{}_graph.txt'.format(opt.args.name)
opt.args.pre_model_save_path = '/data/home/wangchi/model/model_save_pretrain/{}_pretrain.pkl'.format(opt.args.name)
opt.args.final_model_save_path = '/data/home/wangchi/model/model_final/{}_final.pkl'.format(opt.args.name)
os.makedirs('/data/home/wangchi/model/model_final', exist_ok=True)
print("Data: {}".format(opt.args.data_path))
print("Label: {}".format(opt.args.label_path))
print("Data: {}".format(opt.args.data_path))
print("Label: {}".format(opt.args.label_path))

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
adj1, adjn1 = get_adj(adata.X)
adjn1 = adjn1.astype(np.float32)
A1 = coo_matrix(adj1)
X = torch.from_numpy(adata.X)
adjn1 = torch.from_numpy(adjn1)
adjn1 = adjn1.to(device)



dataset = LoadDataset(X)
data = torch.Tensor(dataset.x).to(device)
label = y
args.n_components = Nfeature
args.n_input = Nfeature

###  model definition
model = scDMMGAE(ae_n_enc_1=opt.args.ae_n_enc_1, ae_n_enc_2=opt.args.ae_n_enc_2, ae_n_enc_3=opt.args.ae_n_enc_3,
             ae_n_dec_1=opt.args.ae_n_dec_1, ae_n_dec_2=opt.args.ae_n_dec_2, ae_n_dec_3=opt.args.ae_n_dec_3,
             gae_n_enc_1=opt.args.gae_n_enc_1, gae_n_enc_2=opt.args.gae_n_enc_2,
             gae_n_enc_3=opt.args.gae_n_enc_3, gae_n_enc_4=opt.args.gae_n_enc_4,
             gae_n_dec_1=opt.args.gae_n_dec_1, gae_n_dec_2=opt.args.gae_n_dec_2,
             gae_n_dec_3=opt.args.gae_n_dec_3, gae_n_dec_4=opt.args.gae_n_dec_4,
             n_input=opt.args.n_input,
             n_z=opt.args.n_z,
             n_clusters=opt.args.n_clusters,
             layerd=[20, 128, 1024, Nfeature], hidden=opt.args.n_z, dropout=0.01, n4=Nfeature,
             v=opt.args.freedom_degree,
             n_node=data.size()[0],
             device=device).to(device)

lr = args.lr
Train(opt.args.epoch, model, data, adjn1, label, lr, opt.args.pre_model_save_path,
      opt.args.final_model_save_path,
      opt.args.n_clusters, opt.args.acc, opt.args.gamma_value, opt.args.lambda_value, device)

# 创建figures目录并调用可视化函数

# 创建figures目录
os.makedirs('figures', exist_ok=True)

# 可视化结果
visualize_tsne(opt.args.final_model_save_path, opt.args.name)



