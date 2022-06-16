import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from data_io import ImageDataset, show_images
from networks import CVAELoss, FaceEncoder2, FaceDecoder2


#####   設定情報 <ここから>   #####

# 使用するデバイス
DEVICE = 'cuda:0'

# 全ての訓練データを一回ずつ使用することを「1エポック」として，何エポック分学習するか
N_EPOCHS = 50

# 学習時のバッチサイズ
BATCH_SIZE = 100

# 訓練データセット（画像ファイルリスト）のファイル名
TRAIN_DATA_FILE = './tinyCelebA_list.csv'

# 画像ファイルの先頭に付加する文字列（データセットが存在するディレクトリのパス）
DATA_DIR = './tinyCelebA/'

# 取り扱うラベル
TARGET_LABELS = ['Blond_Hair', 'Chubby', 'Eyeglasses', 'Male', 'Young']

# 画像のサイズ・チャンネル数
C = 3 # チャンネル数
H = 128 # 縦幅
W = 128 # 横幅

# 特徴ベクトルの次元数
N = 128

# 学習結果のニューラルネットワークの保存先
MODEL_FILE_E = './face_cvae_encoder_model.pth'
MODEL_FILE_D = './face_cvae_decoder_model.pth'

#####   設定情報 <ここまで>   #####


### ここから下が処理の本体

# CSVファイルを読み込み，訓練データセットを用意
dataset = ImageDataset(
    filename = TRAIN_DATA_FILE,
    dirname = DATA_DIR,
    input = 'File_Path',
    output = TARGET_LABELS # 読み込むラベルを指定
)

# 訓練データセットを分割し，一部を検証用に回す（高速化のため，全データの 90% は破棄する）
dataset_size = len(dataset)
valid_size = int(0.05 * dataset_size) # 全体の 5% を検証用に
train_size = dataset_size - valid_size # 残りの 95% を学習用に
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

# 訓練データおよび検証用データをミニバッチに分けて使用するための「データローダ」を用意
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ニューラルネットワークの作成
enc_model = FaceEncoder2(C=C, H=H, W=W, N=N, K=len(TARGET_LABELS))
dec_model = FaceDecoder2(C=C, H=H, W=W, N=N, K=len(TARGET_LABELS))
enc_model = enc_model.to(DEVICE)
dec_model = dec_model.to(DEVICE)

# 最適化アルゴリズムの指定（ここでは SGD でなく Adam を使用）
enc_optimizer = optim.Adam(enc_model.parameters())
dec_optimizer = optim.Adam(dec_model.parameters())

# 損失関数
loss_func = CVAELoss()

# 勾配降下法による繰り返し学習
for epoch in range(N_EPOCHS):

    print('Epoch {0}:'.format(epoch + 1))

    # 学習
    enc_model.train()
    dec_model.train()
    sum_loss = 0
    for X, Y in tqdm(train_dataloader):
        for param in enc_model.parameters():
            param.grad = None
        for param in dec_model.parameters():
            param.grad = None
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)
        Z, mu, lnvar = enc_model(X, Y) # 入力値 X を現在のエンコーダに入力し，特徴ベクトル Z を得る
        X_rec = dec_model(Z, Y) # 特徴ベクトル Z を現在のデコーダに入力し，復元画像 Y を得る
        loss = loss_func(X, X_rec, mu, lnvar) # 損失関数の現在値を計算
        loss.backward() # 誤差逆伝播法により，個々のパラメータに関する損失関数の勾配（偏微分）を計算
        enc_optimizer.step() # 勾配に沿ってパラメータの値を更新
        dec_optimizer.step() # 同上
        sum_loss += float(loss) * len(X)
    avg_loss = sum_loss / train_size
    print('train loss = {0:.6f}'.format(avg_loss))

    # 検証
    enc_model.eval()
    dec_model.eval()
    sum_loss = 0
    with torch.inference_mode():
        for X, Y in tqdm(valid_dataloader):
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)
            Z, mu, lnvar = enc_model(X, Y)
            X_rec = dec_model(Z, Y) 
            loss = loss_func(X, X_rec, mu, lnvar)
            sum_loss += float(loss) * len(X)
    avg_loss = sum_loss / valid_size
    if epoch == 0:
        show_images(X.to('cpu').detach(), title='original', save_fig=True)
    show_images(X_rec.to('cpu').detach(), title='epoch {0}'.format(epoch + 1), save_fig=True) # 学習経過の表示
    print('valid loss = {0:.6f}'.format(avg_loss))
    print('')

# 学習結果のニューラルネットワークモデルをファイルに保存
enc_model = enc_model.to('cpu')
dec_model = dec_model.to('cpu')
torch.save(enc_model.state_dict(), MODEL_FILE_E)
torch.save(dec_model.state_dict(), MODEL_FILE_D)
