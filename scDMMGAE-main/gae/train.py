import torch
from opt import args
from utils import eva, get_laplacian, compute_laplacian_loss, multi_distance_loss, compute_embedding_adj
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np

acc_reuslt = []
nmi_result = []
ari_result = []
f1_result = []
use_adjust_lr = []
acc_result = []
nmi_result = []
ari_result = []
ami_result = []  # 新增AMI指标记录


def Pretrain_gae(model, data, adj_list, label, gamma_value):
    """
    使用多种邻接矩阵训练GAE模型
    
    参数:
        model: GAE模型
        data: 输入数据
        adj_list: 不同距离度量下的邻接矩阵列表
        label: 真实标签
        gamma_value: 权重参数
    """
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # 解析距离度量列表
    distance_metrics = args.distance_metrics.split(',')
    
    # 每种度量对应一个邻接矩阵
    assert len(distance_metrics) == len(adj_list), "距离度量数量与邻接矩阵数量不匹配"
    
    # 计算标准化的图拉普拉斯矩阵 (使用第一个邻接矩阵)
    lap = get_laplacian(adj_list[0].to_dense().cpu().numpy())
    lap = torch.FloatTensor(lap).to(adj_list[0].device)
    
    for epoch in range(args.epoch):
        # 使用第一个邻接矩阵进行前向传播
        z_igae, z_hat, adj_hat = model(data, adj_list[0])
        
        loss_w = F.mse_loss(z_hat, torch.spmm(adj_list[0], data))
        loss_a = F.mse_loss(adj_hat, adj_list[0].to_dense())
        
        loss_lap = compute_laplacian_loss(z_igae, lap)
        
        loss_multi_metric = 0.0
        
        for i, metric in enumerate(distance_metrics):
            emb_adj = compute_embedding_adj(z_igae, k=10, metric=metric)
            loss_multi_metric += F.mse_loss(emb_adj, adj_list[i])
        
        loss_multi_metric /= len(distance_metrics)
        
        # 最终损失函数
        loss = loss_w + gamma_value * loss_a + args.lambda_lap * loss_lap + args.lambda_dist * loss_multi_metric
        
        print('{} loss_w: {:.4f}, loss_a: {:.4f}, lap_loss: {:.4f}, multi_metric_loss: {:.4f}'.format(
            epoch, loss_w, loss_a, loss_lap, loss_multi_metric))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 每隔一定轮数评估聚类效果
        if epoch % 5 == 0 or epoch == args.epoch - 1:
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z_igae.data.cpu().numpy())
            nmi, ari, acc, ami, _ = eva(label, kmeans.labels_, epoch, z_igae.data.cpu().numpy())
            
            nmi_result.append(nmi)
            ari_result.append(ari)
            acc_result.append(acc)
            ami_result.append(ami)
        if epoch == args.epoch - 1:
            print(f"Epoch {epoch} -ACC: {np.array(acc_result).max()}, NMI: {np.array(nmi_result).max()}, ARI: {np.array(ari_result).max()},  AMI: {np.array(ami_result).max()}")
        
        torch.save(model.state_dict(), args.model_save_path)

