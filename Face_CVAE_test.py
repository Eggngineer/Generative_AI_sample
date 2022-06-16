import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import copy
import torch
from data_io import show_images
from networks import FaceDecoder2


#####   設定情報 <ここから>   #####

# 使用するデバイス
DEVICE = 'cuda:0'

# 学習時のバッチサイズ
BATCH_SIZE = 100

# 取り扱うラベル
TARGET_LABELS = ['Blond_Hair', 'Chubby', 'Eyeglasses', 'Male', 'Young']

# 画像のサイズ・チャンネル数
C = 3 # チャンネル数
H = 128 # 縦幅
W = 128 # 横幅

# 特徴ベクトルの次元数
N = 128

# 学習済みのニューラルネットワークモデル（デコーダのみ）
MODEL_FILE_D = './face_cvae_decoder_model.pth'

#####   設定情報 <ここまで>   #####


### ここから下が処理の本体

# ニューラルネットワークの作成・学習済みパラメータのロード
dec_model = FaceDecoder2(C=C, H=H, W=W, N=N, K=len(TARGET_LABELS))
dec_model.load_state_dict(torch.load(MODEL_FILE_D))
dec_model = dec_model.to(DEVICE)
dec_model.eval()

# 属性値を固定して乱数ベクトルをランダムに与える場合
attributes = [[0, 0, 0, 0, 1]] # 指定する属性値
Z = torch.randn((BATCH_SIZE, N)).to(DEVICE) # 標準正規分布 N(0,1) に従って適当に乱数ベクトルを作成
Y = torch.tensor(attributes, dtype=torch.float32).repeat((BATCH_SIZE, 1)).to(DEVICE)
X = dec_model(Z, Y) # 乱数ベクトルをデコーダに入力し，その結果を表示
show_images(X.to('cpu').detach(), title='generated', n_data_max=25, n_data_per_row=5, save_fig=True)

'''
# 乱数ベクトルを固定して属性値の一つを変化させる場合
attributes = [0, 0, 0, 0, 1] # ベースとなる属性値
target = 0 # 何番目の属性値を変化させるか
num = 16
Z = torch.randn((1, N)).repeat((num, 1)).to(DEVICE)
Y = []
for i in range(num):
    attributes[target] = i / (num - 1) # 特定の属性値を 0〜1 の範囲でずらす
    Y.append(copy.deepcopy(attributes))
Y = torch.tensor(Y, dtype=torch.float32).to(DEVICE)
X = dec_model(Z, Y) # 乱数ベクトルをデコーダに入力し，その結果を表示
show_images(X.to('cpu').detach(), title='generated', n_data_max=num, n_data_per_row=num, save_fig=True)
'''
