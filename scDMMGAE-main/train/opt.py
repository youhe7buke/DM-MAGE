import argparse

parser = argparse.ArgumentParser(description='scDMMGAE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', type=str, default='Young', choices=['Adam','Muraro','Quake_10x_Bladder','Quake_10x_Limb_Muscle',
             'Quake_10x_Spleen','Quake_Smart-seq2_Diaphragm','Quake_Smart-seq2_Limb_Muscle',
             'Quake_Smart-seq2_Lung','Quake_Smart-seq2_Trachea','Romanov','Young'])
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_z', type=int, default=20)
parser.add_argument('--freedom_degree', type=float, default=1.0)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--active', type=str, default=False)
parser.add_argument('--acc', type=float, default=-1)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--gamma_value', type=float, default=0.1)
parser.add_argument('--lambda_value', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-7)
parser.add_argument('--ae_n_enc_1', type=int, default=1024)
parser.add_argument('--ae_n_enc_2', type=int, default=512)
parser.add_argument('--ae_n_enc_3', type=int, default=128)
parser.add_argument('--ae_n_dec_1', type=int, default=128)
parser.add_argument('--ae_n_dec_2', type=int, default=512)
parser.add_argument('--ae_n_dec_3', type=int, default=1024)
parser.add_argument('--gae_n_enc_1', type=int, default=1024)
parser.add_argument('--gae_n_enc_2', type=int, default=512)
parser.add_argument('--gae_n_enc_3', type=int, default=128)
parser.add_argument('--gae_n_enc_4', type=int, default=20)
parser.add_argument('--gae_n_dec_1', type=int, default=20)
parser.add_argument('--gae_n_dec_2', type=int, default=128)
parser.add_argument('--gae_n_dec_3', type=int, default=512)
parser.add_argument('--gae_n_dec_4', type=int, default=1024)

args = parser.parse_args()