import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import scanpy as sc
import pandas as pd
import seaborn as sns
from GAE import IGAE
from opt import args
from preprocess import prepro, normalize
import os

plt.figure(figsize=(20, 16), dpi=300)

def visualize_tsne(model_path, data_name, save_dir='/data/home/wangchi/scDMMGAE-main/gae/figures'):
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
    x, y = prepro(f'/data/home/wangchi/scDMMGAE/dataset/{data_name}/data.h5')
    
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
    model = IGAE(
        gae_n_enc_1=args.gae_n_enc_1,
        gae_n_enc_2=args.gae_n_enc_2,
        gae_n_enc_3=args.gae_n_enc_3,
        gae_n_enc_4=args.gae_n_enc_4,
        gae_n_dec_1=args.gae_n_dec_1,
        gae_n_dec_2=args.gae_n_dec_2,
        gae_n_dec_3=args.gae_n_dec_3,
        gae_n_dec_4=args.gae_n_dec_4,
        n_input=n_features,
    ).to(device)
    
    # 加载模型参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 准备输入数据
    input_data = torch.FloatTensor(adata.X).to(device)
    
    # 计算图的邻接矩阵（使用模型中相同的预处理方法）
    from utils import get_adj_multi_metric
    distance_metrics = args.distance_metrics.split(',')
    adj_list, adjn_list = get_adj_multi_metric(adata.X, k=10, pca=50, metrics=distance_metrics)
    adjn_tensor = torch.from_numpy(adjn_list[0].astype(np.float32)).to(device)
    
    # 获取嵌入向量
    with torch.no_grad():
        embedding, _, _ = model(input_data, adjn_tensor)
        embedding = embedding.cpu().numpy()
    
    # # 使用t-SNE降维
    # print("计算原始数据的t-SNE...")
    # tsne_original = TSNE(n_components=2, random_state=44).fit_transform(adata.X)
    
    print("计算嵌入向量的t-SNE...")
    tsne_embedding = TSNE(n_components=2, random_state=42).fit_transform(embedding)
    
    # 创建调色板
    n_clusters = int(max(y) - min(y) + 1)
    colors = sns.color_palette("husl", n_colors=n_clusters)
    
    # 可视化原始数据的t-SNE
    plt.figure(figsize=(12, 10))
    # plt.subplot(1, 2, 1)
    # for i in range(n_clusters):
    #     plt.scatter(tsne_original[y == i, 0], tsne_original[y == i, 1], 
    #                c=[colors[i]], label=f'Cluster {i+1}', s=20, alpha=0.7)
    # plt.title('t-SNE', fontsize=16)
    # plt.xlabel('t-SNE 1', fontsize=12)
    # plt.ylabel('t-SNE 2', fontsize=12)
    # plt.legend(markerscale=2)
    
    # # 可视化嵌入向量的t-SNE
    # plt.subplot(1, 2, 2)
    for i in range(n_clusters):
        plt.scatter(tsne_embedding[y == i, 0], tsne_embedding[y == i, 1], 
                   c=[colors[i]], label=f'Cluster {i+1}', s=20, alpha=0.7)
    plt.title(f'{data_name} GAE t-SNE', fontsize=16)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.legend(markerscale=2)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{data_name}_tsne_comparison.png', dpi=300, bbox_inches='tight')
    print(f"可视化结果已保存到 {save_dir}/{data_name}_tsne_comparison.png")
    plt.show()

if __name__ == "__main__":
    # 从命令行参数获取数据集名称，或使用默认值
    data_name = args.name
    model_path = args.model_save_path or f'model/model_save_gae/{data_name}_gae.pkl'
    
    print(f"正在为数据集 {data_name} 生成t-SNE可视化...")
    visualize_tsne(model_path, data_name) 