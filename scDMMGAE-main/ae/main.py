import torch
from AE import AE
import numpy as np
from opt import args
from utils import setup_seed
from torch.utils.data import Dataset, DataLoader
from train import Pretrain_ae
import pandas as pd
import scanpy as sc
from preprocess import prepro, normalize
from utils import get_adj
from scipy.sparse import coo_matrix
import h5py
import opt
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
setup_seed(1)

print("use cuda: {}".format(args.cuda))
device = torch.device("cuda" if args.cuda else "cpu")

def visualize_tsne(model_path, data_name, save_dir='/data/home/wangchi/scDMMGAE-main/ae/figures'):
    """
    使用t-SNE可视化模型的嵌入向量
    
    参数:
        model_path: 模型保存路径
        data_name: 数据集名称
        save_dir: 图像保存目录
    """
    # 创建保存图像的目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    x, y = prepro(f'/data/home/wangchi/scDMMGAE-main/dataset/{data_name}/data.h5')
    
    # 处理某些特定数据集的转置问题
    if data_name in ["klein", "romanov", "Baron", "biase", "goolam", "Xin"]:
        x = x.T
    
    x = np.ceil(x).astype(np.float32)
    y = y.astype(np.float32).flatten()
    
    # 预处理数据
    adata = sc.AnnData(x)
    adata = normalize(adata, filter_min_counts=True, highly_genes=2000, size_factors=True,
                     normalize_input=False, logtrans_input=True)
    
    # 设置设备
    device = torch.device("cuda" if args.cuda else "cpu")
    
    # 初始化模型
    n_features = adata.X.shape[1]
    model = AE(
        ae_n_enc_1=args.ae_n_enc_1,
        ae_n_enc_2=args.ae_n_enc_2,
        ae_n_enc_3=args.ae_n_enc_3,
        ae_n_dec_1=args.ae_n_dec_1,
        ae_n_dec_2=args.ae_n_dec_2,
        ae_n_dec_3=args.ae_n_dec_3,
        n_input=n_features,
        n_z=args.n_z
    ).to(device)
    
    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 准备输入数据
    input_data = torch.FloatTensor(adata.X).to(device)
    
    # 获取嵌入向量
    with torch.no_grad():
        embedding, _ = model(input_data)
        embedding = embedding.cpu().numpy()
    
    print("计算嵌入向量的t-SNE...")
    tsne_embedding = TSNE(n_components=2, random_state=42).fit_transform(embedding)
    
    # 创建调色板
    n_clusters = int(max(y) - min(y) + 1)
    colors = sns.color_palette("husl", n_colors=n_clusters)
    
    # 可视化嵌入向量的t-SNE
    plt.figure(figsize=(12, 10))
    for i in range(n_clusters):
        plt.scatter(tsne_embedding[y == i, 0], tsne_embedding[y == i, 1], 
                   c=[colors[i]], label=f'Cluster {i+1}', s=20, alpha=0.7)
    plt.title(f'{data_name} AE t-SNE', fontsize=16)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.legend(markerscale=2)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{data_name}_tsne.png', dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存到 {save_dir}/{data_name}_tsne.png")
    plt.show()

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


args.model_save_path = 'model/model_save_ae/{}_ae.pkl'.format(args.name)
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
adj1, adjn1 = get_adj(adata.X)
A1 = coo_matrix(adj1)
X = torch.from_numpy(adata.X)
adjn1 = torch.from_numpy(adjn1)

dataset = LoadDataset(X)
train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
args.n_input = Nfeature

model = AE(
    ae_n_enc_1=args.ae_n_enc_1,
    ae_n_enc_2=args.ae_n_enc_2,
    ae_n_enc_3=args.ae_n_enc_3,
    ae_n_dec_1=args.ae_n_dec_1,
    ae_n_dec_2=args.ae_n_dec_2,
    ae_n_dec_3=args.ae_n_dec_3,
    n_input=args.n_input,
    n_z=args.n_z).to(device)

Pretrain_ae(model, dataset, y, train_loader, device)

# 在训练完成后添加可视化
if args.visualize:
    print("正在生成t-SNE可视化...")
    visualize_tsne(args.model_save_path, args.name)