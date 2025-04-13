import argparse




parser = argparse.ArgumentParser(description='IGAE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', type=str, default='Young', choices=['Adam','Muraro','Quake_10x_Bladder','Quake_10x_Limb_Muscle',
             'Quake_10x_Spleen','Quake_Smart-seq2_Diaphragm','Quake_Smart-seq2_Limb_Muscle',
             'Quake_Smart-seq2_Lung','Quake_Smart-seq2_Trachea','Romanov','Young'])
parser.add_argument('--lr', type=float, default=5e-7)
parser.add_argument('--k', type=int, default=None)
parser.add_argument('--n_clusters', type=int, default=11)
parser.add_argument('--n_z', type=int, default=10)
parser.add_argument('--n_input', type=int, default=2000)
parser.add_argument('--gamma_value', type=float, default=0.1)
parser.add_argument('--data_path', type=str, default='.txt')
parser.add_argument('--label_path', type=str, default='.txt')
parser.add_argument('--save_path', type=str, default='.txt')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--n_components', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--acc', type=float, default=-1)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--gae_n_enc_1', type=int, default=1024)
parser.add_argument('--gae_n_enc_2', type=int, default=512)
parser.add_argument('--gae_n_enc_3', type=int, default=128)
parser.add_argument('--gae_n_enc_4', type=int, default=20)
parser.add_argument('--gae_n_dec_1', type=int, default=20)
parser.add_argument('--gae_n_dec_2', type=int, default=128)
parser.add_argument('--gae_n_dec_3', type=int, default=512)
parser.add_argument('--gae_n_dec_4', type=int, default=1024)
parser.add_argument('--lambda_lap', type=float, default=20,
                   help='Weight for Laplacian embedding loss')
parser.add_argument('--distance_metrics', type=str, default='euclidean,cosine',
                   help='Distance metrics to use, comma-separated')
parser.add_argument('--lambda_dist', type=float, default=1,
                   help='Weight for multi-distance loss')
parser.add_argument('--lambda_adj_consist', type=float, default=0.5,
                   help='Weight for adjacency matrix consistency loss')
parser.add_argument('--model_save_path', type=str, default=None,
                   help='Path to save model checkpoint')
parser.add_argument('--visualize', action='store_false', help='是否在训练后生成t-SNE可视化')

args = parser.parse_args()