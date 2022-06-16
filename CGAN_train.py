import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from data_io import ImageDataset, show_images
from networks import Generator2, Discriminator2


#####   設定情報 <ここから>   #####

# 使用するデバイス
DEVICE = 'cuda:0'

# 全ての訓練データを一回ずつ使用することを「1エポック」として，何エポック分学習するか
N_EPOCHS = 500

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
N = 32

# 学習結果のニューラルネットワークの保存先
MODEL_FILE_G = './face_cgan_generator_model.pth'
MODEL_FILE_D = './face_cgan_discriminator_model.pth'

#####   設定情報 <ここまで>   #####


### ここから下が処理の本体

# CSVファイルを読み込み，訓練データセットを用意
dataset = ImageDataset(
    filename = TRAIN_DATA_FILE,
    dirname = DATA_DIR,
    input = 'File_Path',
    output = TARGET_LABELS # 読み込むラベルを指定
)

# 訓練データセットを分割し，一方を検証用に回す
dataset_size = len(dataset)
valid_size = int(0.05 * dataset_size) # 全体の 5% を検証用に
train_size = dataset_size - valid_size # 残りの 95% を学習用に
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

# 訓練データおよび検証用データをミニバッチに分けて使用するための「データローダ」を用意
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ニューラルネットワークの作成
gen_model = Generator2(C=C, H=H, W=W, N=N, K=len(TARGET_LABELS))
dis_model = Discriminator2(C=C, H=H, W=W, N=N, K=len(TARGET_LABELS))
gen_model = gen_model.to(DEVICE)
dis_model = dis_model.to(DEVICE)

# 最適化アルゴリズムの指定（ここでは SGD でなく Adam を使用）
gen_optimizer = optim.Adam(gen_model.parameters(), lr=0.0002, betas=(0.5, 0.999))
dis_optimizer = optim.Adam(dis_model.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 損失関数：平均二乗損失を使用
loss_func = nn.BCELoss()

# 勾配降下法による繰り返し学習
for epoch in range(N_EPOCHS):

    print('Epoch {0}:'.format(epoch + 1))

    # 学習
    gen_model.train()
    dis_model.train()
    sum_gen_loss = 0
    sum_dis_loss = 0
    for X, L in tqdm(train_dataloader):
        for param in gen_model.parameters():
            param.grad = None
        for param in dis_model.parameters():
            param.grad = None
        X = X.to(DEVICE)
        L = L.to(DEVICE)
        L_img = L.reshape(*L.size(), 1, 1).repeat(1, 1, X.size()[2], X.size()[3]) # 属性情報を画像と同じ形に拡張
        Z = torch.randn(len(X), N).to(DEVICE) # 乱数生成
        fake = gen_model(Z, L) # Fake画像を生成
        fake_cpy = fake.detach() # Fake画像のコピーを用意しておく
        ones = torch.ones(len(X), 1).to(DEVICE) # 損失関数計算用の定数
        zeros = torch.zeros(len(X), 1).to(DEVICE) # 損失関数計算用の定数
        ### Generator の学習 ###
        Y_fake = dis_model(fake, L_img) # Fake画像を識別
        gen_loss = loss_func(Y_fake, ones)
        gen_loss.backward()
        gen_optimizer.step()
        ### Discriminator の学習 ###
        for param in dis_model.parameters():
            param.grad = None # Generatorの学習のために計算した勾配を一旦リセット
        Y_real = dis_model(X, L_img) # Real画像を識別
        Y_fake_cpy = dis_model(fake_cpy, L_img) # Fake画像を識別
        dis_loss = loss_func(Y_fake_cpy, zeros) + loss_func(Y_real, ones)
        dis_loss.backward()
        dis_optimizer.step()
        ### 損失関数の値を出力用変数に退避 ###
        sum_gen_loss += float(gen_loss) * len(X)
        sum_dis_loss += float(dis_loss) * len(X)
    avg_gen_loss = sum_gen_loss / train_size
    avg_dis_loss = sum_dis_loss / train_size
    print('generator train loss = {0:.6f}'.format(avg_gen_loss))
    print('discriminator train loss = {0:.6f}'.format(avg_dis_loss))

    # 検証
    gen_model.eval()
    dis_model.eval()
    with torch.inference_mode():
        Z = torch.randn(BATCH_SIZE, N).to(DEVICE) # 乱数生成
        L = torch.zeros(BATCH_SIZE, len(TARGET_LABELS)).to(DEVICE)
        #L = torch.randint(0, 2, (len(X), len(TARGET_LABELS))).to(DEVICE, dtype=torch.float32) # fake用の属性も乱数で生成
        fake = gen_model(Z, L) # Fake画像を生成
    show_images(fake.to('cpu').detach(), title='epoch {0}'.format(epoch + 1), n_data_max=64, n_data_per_row=8, save_fig=True) # 学習経過の表示
    print('')

    # 現時点での生成器のネットワークモデルをファイルに保存
    if (epoch + 1) % 10 == 0:
        gen_model = gen_model.to(DEVICE)
        torch.save(gen_model.state_dict(), './face_cgan_generator_model_ep{0}.pth'.format(epoch + 1))
        gen_model = gen_model.to(DEVICE)

# 学習結果のニューラルネットワークモデルをファイルに保存
gen_model = gen_model.to('cpu')
dis_model = dis_model.to('cpu')
torch.save(gen_model.state_dict(), MODEL_FILE_G)
torch.save(dis_model.state_dict(), MODEL_FILE_D)
