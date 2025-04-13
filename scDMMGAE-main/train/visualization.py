import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scanpy as sc
import pandas as pd
import seaborn as sns
from scDMMGAE import scDMMGAE
from opt import args
from preprocess import prepro, normalize
import os

plt.figure(figsize=(20, 16), dpi=300)

def visualize_tsne(model_path, data_name, save_dir='/data/home/wangchi/scDMMGAE-main/train/figures'):
    """
    使用t-SNE可视化模型的嵌入向量并与原始数据进行对比
    
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
    
    # 初始化模型，提供所有必要的参数
    n_features = adata.X.shape[1]
    n_clusters = int(max(y) - min(y) + 1)
    n_samples = adata.X.shape[0]
    
    model = scDMMGAE(
        ae_n_enc_1=args.ae_n_enc_1, 
        ae_n_enc_2=args.ae_n_enc_2, 
        ae_n_enc_3=args.ae_n_enc_3,
        ae_n_dec_1=args.ae_n_dec_1, 
        ae_n_dec_2=args.ae_n_dec_2, 
        ae_n_dec_3=args.ae_n_dec_3,
        gae_n_enc_1=args.gae_n_enc_1, 
        gae_n_enc_2=args.gae_n_enc_2,
        gae_n_enc_3=args.gae_n_enc_3, 
        gae_n_enc_4=args.gae_n_enc_4,
        gae_n_dec_1=args.gae_n_dec_1, 
        gae_n_dec_2=args.gae_n_dec_2,
        gae_n_dec_3=args.gae_n_dec_3, 
        gae_n_dec_4=args.gae_n_dec_4,
        n_input=n_features,
        n_z=args.n_z,
        n_clusters=n_clusters,
        layerd=[20, 128, 1024, n_features], 
        hidden=args.n_z, 
        dropout=0.01, 
        n4=n_features,
        v=args.freedom_degree,
        n_node=n_samples,
        device=device
    ).to(device)
    
    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 准备输入数据
    input_data = torch.FloatTensor(adata.X).to(device)
    
    # 计算图的邻接矩阵
    from utils import get_adj
    adj, adjn = get_adj(adata.X)
    adjn = adjn.astype(np.float32)
    adjn_tensor = torch.from_numpy(adjn).to(device)
    
    # 获取嵌入向量
    with torch.no_grad():
        z = model.ae.encoder(input_data)
        z_g, _ = model.gae.encoder(input_data, adjn_tensor)
        embedding = torch.cat((z, z_g), dim=1).cpu().numpy()
    
    # 使用t-SNE降维
    print("计算嵌入向量的t-SNE...")
    tsne_embedding = TSNE(n_components=2, random_state=42).fit_transform(embedding)
    
    # 创建调色板
    colors = sns.color_palette("husl", n_colors=n_clusters)
    
    # 可视化嵌入向量的t-SNE
    plt.figure(figsize=(12, 10))
    for i in range(n_clusters):
        plt.scatter(tsne_embedding[y == i, 0], tsne_embedding[y == i, 1], 
                   c=[colors[i]], label=f'Cluster {i+1}', s=20, alpha=0.7)
    plt.title(f'{data_name} scDMMGAE t-SNE', fontsize=16)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.legend(markerscale=2)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{data_name}_tsne_scDMMGAE.png', dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存到 {save_dir}/{data_name}_tsne_scDMMGAE.png")
    plt.close()

if __name__ == "__main__":
    # 从命令行参数获取数据集名称，或使用默认值
    data_name = args.name
    model_path = args.model_save_path or f'model/model_save/{data_name}_model.pkl'
    
    print(f"正在为数据集 {data_name} 生成t-SNE可视化...")
    visualize_tsne(model_path, data_name) 