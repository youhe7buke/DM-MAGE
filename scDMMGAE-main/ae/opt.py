import argparse

parser = argparse.ArgumentParser(description='AE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', type=str, default='Young',choices=['Adam','Muraro','Quake_10x_Bladder','Quake_10x_Limb_Muscle',
             'Quake_10x_Spleen','Quake_Smart-seq2_Diaphragm','Quake_Smart-seq2_Limb_Muscle',
             'Quake_Smart-seq2_Lung','Quake_Smart-seq2_Trachea','Romanov','Young'])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_clusters', type=int, default=10000)
parser.add_argument('--n_z', type=int, default=20)
parser.add_argument('--n_input', type=int, default=2000)
parser.add_argument('--data_path', type=str, default='.txt')
parser.add_argument('--label_path', type=str, default='.txt')
parser.add_argument('--save_path', type=str, default='.txt')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--n_components', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--acc', type=float, default=-1)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--ae_n_enc_1', type=int, default=1024)
parser.add_argument('--ae_n_enc_2', type=int, default=512)
parser.add_argument('--ae_n_enc_3', type=int, default=128)
parser.add_argument('--ae_n_dec_1', type=int, default=128)
parser.add_argument('--ae_n_dec_2', type=int, default=512)
parser.add_argument('--ae_n_dec_3', type=int, default=1024)
parser.add_argument('--model_save_path', type=str, default=None,
                   help='Path to save model checkpoint')
parser.add_argument('--visualize', action='store_false', help='是否在训练后生成t-SNE可视化')

args = parser.parse_args()