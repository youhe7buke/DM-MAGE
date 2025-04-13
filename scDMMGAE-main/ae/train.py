import torch
from opt import args
from utils import eva
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.cluster import KMeans

acc_reuslt = []
nmi_result = []
ari_result = []
f1_result = []
use_adjust_lr = []


def Pretrain_ae(model, dataset, y, train_loader, device):
    optimizer = Adam(model.parameters(), lr=args.lr)
    best_nmi = 0  # 记录最佳NMI分数
    
    for epoch in range(args.epoch):
        # if (args.name in use_adjust_lr):
        #     adjust_learning_rate(optimizer, epoch)
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            x = torch.Tensor(dataset.x).to(device).float()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            print('{} loss: {}'.format(epoch, loss))

            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z.data.cpu().numpy())

            nmi, ari, acc, ami, sil = eva(y, kmeans.labels_, epoch, z.data.cpu().numpy())
            
            # 只在得到更好的结果时保存模型
            if nmi > best_nmi:
                best_nmi = nmi
                torch.save(model.state_dict(), args.model_save_path)
                print(f"发现更好的模型(NMI={nmi:.4f})，已保存到 {args.model_save_path}")
            
            nmi_result.append(nmi)
            ari_result.append(ari)
