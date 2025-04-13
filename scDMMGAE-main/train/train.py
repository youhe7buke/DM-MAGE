import opt
import torch
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
from utils import adjust_learning_rate
from utils import eva, target_distribution
from ZINB import ZINB
from tensorboardX import SummaryWriter, writer

acc_reuslt = []
nmi_result = []
ari_result = []
ami_result = []
sil_result = []
f1_result = []
use_adjust_lr = []

writer = SummaryWriter('logs1')

def Train(epochs, model, data, adj, label, lr, pre_model_save_path, final_model_save_path, n_clusters,
          original_acc, gamma_value, lambda_value, device):
    optimizer = Adam(model.parameters(), lr=lr)
    model.load_state_dict(torch.load(pre_model_save_path, map_location=device))  #_pretrain.pkl
    with torch.no_grad():
        x_hat, z_hat, adj_hat, z_ae, z_igae, _, _, _, z_tilde, _, _, _ = model(data, adj)
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    cluster_id = kmeans.fit_predict(z_tilde.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    # 初始化评估
    eva(label, cluster_id, 'Initialization', features=z_tilde.data.cpu().numpy())
    
    best_nmi = 0  # 记录最佳NMI分数

    for epoch in range(epochs):
        # if opt.args.name in use_adjust_lr:
        #     adjust_learning_rate(optimizer, epoch)
        x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde, pi, disp, mean  = model(data, adj)

        tmp_q = q.data
        p = target_distribution(tmp_q)
        zinb = ZINB(pi, theta=disp, ridge_lambda=0)
        loss_zinb = zinb.loss(data, mean, mean=True)
        loss_zinb = loss_zinb.double()
        loss_ae = F.mse_loss(x_hat, data)
        loss_w = F.mse_loss(z_hat, torch.spmm(adj, data))
        loss_a = F.mse_loss(adj_hat, adj.to_dense())
        loss_igae = loss_w + gamma_value * loss_a
        loss_kl = F.kl_div((q.log() + q1.log() + q2.log()) / 3, p, reduction='batchmean')
        loss = 0.3*loss_ae + 0.3*loss_igae + 0.3*lambda_value * loss_kl + 0.1*loss_zinb

        if (epoch+1) % 10 == 0:
            print('{} loss: {}'.format(epoch, loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        kmeans = KMeans(n_clusters=n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy())

        # 使用新的评价指标函数
        nmi, ari, acc, ami, sil = eva(label, kmeans.labels_, epoch, features=z_tilde.data.cpu().numpy())
        
        acc_reuslt.append(acc)
        nmi_result.append(nmi)
        ari_result.append(ari)
        ami_result.append(ami)
        if sil is not None:
            sil_result.append(sil)

        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('acc', acc, epoch)
        writer.add_scalar('nmi', nmi, epoch)
        writer.add_scalar('ari', ari, epoch)
        writer.add_scalar('ami', ami, epoch)
        if sil is not None:
            writer.add_scalar('silhouette', sil, epoch)
        
        # 只在得到更好的NMI时保存模型
        if nmi > best_nmi:
            best_nmi = nmi
            torch.save(model.state_dict(), final_model_save_path)
            print(f"发现更好的模型(NMI={nmi:.4f})，已保存到 {final_model_save_path}")
    
    # 打印最佳结果
    print("最佳结果:")
    print("ACC: {:.4f}".format(max(acc_reuslt)))
    print("NMI: {:.4f}".format(max(nmi_result)))
    print("ARI: {:.4f}".format(max(ari_result)))
    print("AMI: {:.4f}".format(max(ami_result)))
    if len(sil_result) > 0:
        print("Silhouette: {:.4f}".format(max(sil_result)))
        
    writer.close()
