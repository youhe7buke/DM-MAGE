import torch
from opt import args
from utils import eva
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans
from ZINB import ZINB
acc_reuslt = []
nmi_result = []
ari_result = []
ami_result = []
sil_result = []
f1_result = []
use_adjust_lr = []


def Pretrain(model, data, adj, label):
    optimizer = Adam(model.parameters(), lr=args.lr)
    best_nmi = 0  # 记录最佳NMI分数
    
    for epoch in range(args.epoch):
        # if (args.name in use_adjust_lr):
        #     adjust_learning_rate(optimizer, epoch)

        x_hat, z_hat, adj_hat, z_ae, z_igae, z_tilde, pi, disp, mean = model(data, adj)
        zinb = ZINB(pi, theta=disp, ridge_lambda=0)
        loss_zinb = zinb.loss(data, mean, mean=True)
        loss_zinb = loss_zinb.double()
        loss_1 = F.mse_loss(x_hat, data)
        loss_2 = F.mse_loss(z_hat, torch.spmm(adj, data))
        loss_3 = F.mse_loss(adj_hat, adj.to_dense())
        loss_4 = F.mse_loss(z_ae, z_igae)  # simple aligned

        loss =0.3* (loss_1 + args.alpha * loss_2 + args.beta \
               * loss_3 + args.omega * loss_4 )+ 0.1*loss_zinb# you can tune all kinds of hyper-parameters to get better performance.

        print('{} loss: {}'.format(epoch, loss), '{} loss: {}'.format(epoch, loss_zinb))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z_tilde.data.cpu().numpy())

        # 使用新的评价指标函数
        nmi, ari, acc, ami, sil = eva(label, kmeans.labels_, epoch, features=z_tilde.data.cpu().numpy())
        acc_reuslt.append(acc)
        nmi_result.append(nmi)
        ari_result.append(ari)
        ami_result.append(ami)
        if sil is not None:
            sil_result.append(sil)
            
        # 只在得到更好的NMI时保存模型
        if nmi > best_nmi:
            best_nmi = nmi
            torch.save(model.state_dict(), args.pre_model_save_path)
            print(f"发现更好的模型(NMI={nmi:.4f})，已保存到 {args.pre_model_save_path}")
        
    # 打印最佳结果
    print("最佳结果:")
    print("ACC: {:.4f}".format(max(acc_reuslt)))
    print("NMI: {:.4f}".format(max(nmi_result)))
    print("ARI: {:.4f}".format(max(ari_result)))
    print("AMI: {:.4f}".format(max(ami_result)))
    if len(sil_result) > 0:
        print("Silhouette: {:.4f}".format(max(sil_result)))
