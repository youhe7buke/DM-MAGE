import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from AE import AE
from IGAE import IGAE
from ZINB import decoder_ZINB

class scDMMGAE(nn.Module):

    def __init__(self, ae_n_enc_1, ae_n_enc_2, ae_n_enc_3,
                 ae_n_dec_1, ae_n_dec_2, ae_n_dec_3,
                 gae_n_enc_1, gae_n_enc_2, gae_n_enc_3,gae_n_enc_4,
                 gae_n_dec_1, gae_n_dec_2, gae_n_dec_3,gae_n_dec_4,
                 n_input, n_z, n_clusters, layerd, hidden, dropout,n4=100,v=1.0, n_node=None, device=None):
        super(scDMMGAE, self).__init__()

        self.ae = AE(
            ae_n_enc_1=ae_n_enc_1,
            ae_n_enc_2=ae_n_enc_2,
            ae_n_enc_3=ae_n_enc_3,
            ae_n_dec_1=ae_n_dec_1,
            ae_n_dec_2=ae_n_dec_2,
            ae_n_dec_3=ae_n_dec_3,
            n_input=n_input,
            n_z=n_z)

        self.gae = IGAE(
            gae_n_enc_1=gae_n_enc_1,
            gae_n_enc_2=gae_n_enc_2,
            gae_n_enc_3=gae_n_enc_3,
            gae_n_enc_4=gae_n_enc_4,
            gae_n_dec_1=gae_n_dec_1,
            gae_n_dec_2=gae_n_dec_2,
            gae_n_dec_3=gae_n_dec_3,
            gae_n_dec_4=gae_n_dec_4,
            n_input=n_input)
        self.Decoder = decoder_ZINB(layerd, hidden, n4, dropout)
        #layerd = [20,]
        #fusion parameter
        self.a = nn.Parameter(nn.init.constant_(torch.zeros(n_node, n_z), 0.5), requires_grad=True).to(device)
        self.b = 1 - self.a

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, n_z), requires_grad=True)
        torch.nn.init.xavier_normal_(self.cluster_layer.data)#xavier的初始化方式和BN一样，为了保证数据的分布（均值方差一致）是一样的，加快收敛，就这么简单吧。

        self.v = v
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x, adj):
        z_ae = self.ae.encoder(x)
        z_igae, z_igae_adj = self.gae.encoder(x, adj)
        z_i = self.a * z_ae + self.b * z_igae
        z_l = torch.spmm(adj, z_i)
        s = torch.mm(z_l, z_l.t())
        s = F.softmax(s, dim=1)  #对n维输入张量运用Softmax函数，将张量的每个元素缩放到（0,1）区间且和为1  行归一化

        z_g = torch.mm(s, z_l)
        z_tilde = self.gamma * z_g + z_l
        pi, disp, mean = self.Decoder(z_tilde)
        self.mean = mean

        x_hat = self.ae.decoder(z_tilde)
        z_hat, z_hat_adj = self.gae.decoder(z_tilde, adj)
        adj_hat = z_igae_adj + z_hat_adj

        q = 1.0 / (1.0 + torch.sum(torch.pow((z_tilde).unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        #unsqueeze在第一个维度(中括号)的每个元素加中括号
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        q1 = 1.0 / (1.0 + torch.sum(torch.pow(z_ae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q1 = q1.pow((self.v + 1.0) / 2.0)
        q1 = (q1.t() / torch.sum(q1, 1)).t()

        q2 = 1.0 / (1.0 + torch.sum(torch.pow(z_igae.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q2 = q2.pow((self.v + 1.0) / 2.0)
        q2 = (q2.t() / torch.sum(q2, 1)).t()

        return x_hat, z_hat, adj_hat, z_ae, z_igae, q, q1, q2, z_tilde, pi, disp, mean
