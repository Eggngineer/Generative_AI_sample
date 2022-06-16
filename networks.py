import torch
import torch.nn as nn
import torch.nn.functional as F


# VAE用の損失関数
class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
    def forward(self, x, y, mu, lnvar):
        kl = -0.5 * torch.sum(1 + lnvar - mu**2 - torch.exp(lnvar))
        return F.binary_cross_entropy(y, x, reduction='sum') + kl


# 顔画像CVAE用の損失関数
class CVAELoss(nn.Module):
    def __init__(self):
        super(CVAELoss, self).__init__()
    def forward(self, x_input, x_rec, mu, lnvar):
        rec = torch.mean(torch.abs(x_rec - x_input))
        kl = -0.5 * torch.mean(1 + lnvar - mu**2 - torch.exp(lnvar))
        return rec + 0.1 * kl


# Residual Block
class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, activation=F.relu):
        super(ResBlock, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.bn2 = nn.BatchNorm2d(num_features=channels)
    def forward(self, x):
        h = self.activation(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.activation(h + x)


# MNIST画像を N 次元の特徴ベクトルへと圧縮するニューラルネットワーク
# AutoEncoderのエンコーダ部分のサンプル
class MNISTEncoder(nn.Module):

    # N: 出力の特徴ベクトルの次元数
    def __init__(self, N, use_BatchNorm=False):
        super(MNISTEncoder, self).__init__()

        # 畳込み層1,2
        # カーネルサイズ4，ストライド幅2，パディング1の設定なので，これらを通すことにより特徴マップの縦幅・横幅がそれぞれ 1/2 になる
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2, padding=1)

        # 畳込み層3
        # カーネルサイズ3，ストライド幅1，パディング1の設定なので，これを通しても特徴マップの縦幅・横幅は変化しない
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)

        # バッチ正規化層
        self.use_BatchNorm = use_BatchNorm
        if use_BatchNorm:
            self.bn1 = nn.BatchNorm2d(num_features=4) # num_features は conv1 の out_channels と同じ値に
            self.bn2 = nn.BatchNorm2d(num_features=8) # num_features は conv2 の out_channels と同じ値に
            self.bn3 = nn.BatchNorm2d(num_features=8) # num_features は conv3 の out_channels と同じ値に

        # 平坦化
        self.flat = nn.Flatten()

        # 全結合層
        # 畳込み層1, 2を通すことにより特徴マップの縦幅・横幅は 28/4 = 7 になっている．
        # その後，さらに self.conv3 を通してから全結合層を適用する予定なので，入力側のパーセプトロン数は 8*7*7
        self.fc = nn.Linear(in_features=8*7*7, out_features=N)

    def forward(self, x):
        if self.use_BatchNorm:
            h = F.leaky_relu(self.bn1(self.conv1(x)))
            h = F.leaky_relu(self.bn2(self.conv2(h)))
            h = F.leaky_relu(self.bn3(self.conv3(h)))
        else:
            h = F.leaky_relu(self.conv1(x))
            h = F.leaky_relu(self.conv2(h))
            h = F.leaky_relu(self.conv3(h))
        h = self.flat(h)
        z = self.fc(h)
        return z


# MNIST画像を N 次元の特徴ベクトルへと圧縮するニューラルネットワーク（VAE版）
# VAEのエンコーダ部分のサンプル
class MNISTEncoder2(nn.Module):

    # N: 出力の特徴ベクトルの次元数
    def __init__(self, N, use_BatchNorm=False):
        super(MNISTEncoder2, self).__init__()

        # 畳込み層1,2
        # カーネルサイズ4，ストライド幅2，パディング1の設定なので，これらを通すことにより特徴マップの縦幅・横幅がそれぞれ 1/2 になる
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=2, padding=1)

        # 畳込み層3
        # カーネルサイズ3，ストライド幅1，パディング1の設定なので，これを通しても特徴マップの縦幅・横幅は変化しない
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)

        # バッチ正規化層
        self.use_BatchNorm = use_BatchNorm
        if use_BatchNorm:
            self.bn1 = nn.BatchNorm2d(num_features=4) # num_features は conv1 の out_channels と同じ値に
            self.bn2 = nn.BatchNorm2d(num_features=8) # num_features は conv2 の out_channels と同じ値に
            self.bn3 = nn.BatchNorm2d(num_features=8) # num_features は conv3 の out_channels と同じ値に

        # 平坦化
        self.flat = nn.Flatten()

        # 全結合層
        # 畳込み層1, 2を通すことにより特徴マップの縦幅・横幅は 28/4 = 7 になっている．
        # その後，さらに self.conv3 を通してから全結合層を適用する予定なので，入力側のパーセプトロン数は 8*7*7
        self.fc_mu = nn.Linear(in_features=8*7*7, out_features=N)
        self.fc_lnvar = nn.Linear(in_features=8*7*7, out_features=N)

    def forward(self, x):
        if self.use_BatchNorm:
            h = F.leaky_relu(self.bn1(self.conv1(x)))
            h = F.leaky_relu(self.bn2(self.conv2(h)))
            h = F.leaky_relu(self.bn3(self.conv3(h)))
        else:
            h = F.leaky_relu(self.conv1(x))
            h = F.leaky_relu(self.conv2(h))
            h = F.leaky_relu(self.conv3(h))
        h = self.flat(h)
        mu = self.fc_mu(h)
        lnvar = self.fc_lnvar(h)
        eps = torch.randn_like(mu) # mu と同じサイズの標準正規乱数を生成
        z = mu + eps * torch.exp(0.5 * lnvar)
        return z, mu, lnvar


# N 次元の特徴ベクトルからMNIST風画像を生成するニューラルネットワーク
# AutoEncoderのデコーダ部分のサンプル
class MNISTDecoder(nn.Module):

    # N: 出力の特徴ベクトルの次元数
    def __init__(self, N, use_BatchNorm=False):
        super(MNISTDecoder, self).__init__()

        # 全結合層
        # パーセプトロン数は MNISTEncoder の全結合層と真逆に設定
        self.fc = nn.Linear(in_features=N, out_features=8*7*7)

        # 逆畳込み層1～3
        # カーネルサイズ，ストライド幅，パディングは MNISTEncoder の畳込み層1～3と真逆に設定
        self.deconv3 = nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=4, stride=2, padding=1)

        # 畳込み層
        # 逆畳込み層の出力には checker board artifact というノイズが乗りやすいので，最後に畳込み層を通しておく
        self.conv = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1)

        # バッチ正規化層
        self.use_BatchNorm = use_BatchNorm
        if use_BatchNorm:
            self.bn3 = nn.BatchNorm2d(num_features=8)
            self.bn2 = nn.BatchNorm2d(num_features=4)
            self.bn1 = nn.BatchNorm2d(num_features=4)

    def forward(self, z):
        h = F.leaky_relu(self.fc(z))
        h = torch.reshape(h, (len(h), 8, 7, 7)) # 一列に並んだパーセプトロンを 8*7*7 の特徴マップに並べ直す
        if self.use_BatchNorm:
            h = F.leaky_relu(self.bn3(self.deconv3(h)))
            h = F.leaky_relu(self.bn2(self.deconv2(h)))
            h = F.leaky_relu(self.bn1(self.deconv1(h)))
        else:
            h = F.leaky_relu(self.deconv3(h))
            h = F.leaky_relu(self.deconv2(h))
            h = F.leaky_relu(self.deconv1(h))
        y = torch.sigmoid(self.conv(h))
        return y


# 顔画像を N 次元の特徴ベクトルへと圧縮するニューラルネットワーク
# AutoEncoderのエンコーダ部分のサンプル
class FaceEncoder(nn.Module):

    # C: 入力顔画像のチャンネル数（1または3と仮定）
    # H: 入力顔画像の縦幅（8の倍数と仮定）
    # W: 入力顔画像の横幅（8の倍数と仮定）
    # N: 出力の特徴ベクトルの次元数
    def __init__(self, C, H, W, N):
        super(FaceEncoder, self).__init__()

        # 畳込み層1～3
        # カーネルサイズ4，ストライド幅2，パディング1の設定なので，これらを通すことにより特徴マップの縦幅・横幅がそれぞれ 1/2 になる
        self.conv1 = nn.Conv2d(in_channels=C, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)

        # 畳込み層4
        # カーネルサイズ3，ストライド幅1，パディング1の設定なので，これを通しても特徴マップの縦幅・横幅は変化しない
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        #self.conv4 = ResBlock(channels=64, kernel_size=3, stride=1, padding=1) # 例: 15行目で定義した Residual Block を使用する場合はこのように記載

        # バッチ正規化層
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.bn4 = nn.BatchNorm2d(num_features=64)

        # 平坦化
        self.flat = nn.Flatten()

        # 全結合層1
        # 畳込み層1～3を通すことにより特徴マップの縦幅・横幅は都合 1/8 になっている．
        # その後，さらに self.conv4 を通してから全結合層を適用する予定なので，入力側のパーセプトロン数は 64*(H/8)*(W/8) = H*W
        self.fc1 = nn.Linear(in_features=H*W, out_features=2048)

        # 全結合層2
        self.fc2 = nn.Linear(in_features=2048, out_features=N)

    def forward(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)))
        h = F.leaky_relu(self.bn2(self.conv2(h)))
        h = F.leaky_relu(self.bn3(self.conv3(h)))
        h = F.leaky_relu(self.bn4(self.conv4(h)))
        h = self.flat(h)
        h = F.leaky_relu(self.fc1(h))
        z = self.fc2(h)
        return z


# N 次元の特徴ベクトルから顔画像を生成するニューラルネットワーク
# AutoEncoderのデコーダ部分のサンプル
class FaceDecoder(nn.Module):

    # C: 入力顔画像のチャンネル数（1または3と仮定）
    # H: 入力顔画像の縦幅（8の倍数と仮定）
    # W: 入力顔画像の横幅（8の倍数と仮定）
    # N: 出力の特徴ベクトルの次元数
    def __init__(self, C, H, W, N):
        super(FaceDecoder, self).__init__()
        self.W = W
        self.H = H

        # 全結合層1,2
        # パーセプトロン数は FaceEncoder の全結合層と真逆に設定
        self.fc2 = nn.Linear(in_features=N, out_features=2048)
        self.fc1 = nn.Linear(in_features=2048, out_features=H*W)

        # 逆畳込み層1～4
        # カーネルサイズ，ストライド幅，パディングは FaceEncoder の畳込み層1～4と真逆に設定
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)

        # バッチ正規化層
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.bn1 = nn.BatchNorm2d(num_features=8)

        # 畳込み層
        # 逆畳込み層の出力には checker board artifact というノイズが乗りやすいので，最後に畳込み層を通しておく
        self.conv = nn.Conv2d(in_channels=8, out_channels=C, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = F.leaky_relu(self.fc2(z))
        h = F.leaky_relu(self.fc1(h))
        h = torch.reshape(h, (len(h), 64, self.H//8, self.W//8)) # 一列に並んだパーセプトロンを 64*(H/8)*(W/8) の特徴マップに並べ直す
        h = F.leaky_relu(self.bn4(self.deconv4(h)))
        h = F.leaky_relu(self.bn3(self.deconv3(h)))
        h = F.leaky_relu(self.bn2(self.deconv2(h)))
        h = F.leaky_relu(self.bn1(self.deconv1(h)))
        y = torch.sigmoid(self.conv(h))
        return y


# 顔画像を N 次元の特徴ベクトルへと圧縮するニューラルネットワーク（CVAE版）
# CVAEのエンコーダ部分のサンプル
class FaceEncoder2(nn.Module):

    # C: 入力顔画像のチャンネル数（1または3と仮定）
    # H: 入力顔画像の縦幅（16の倍数と仮定）
    # W: 入力顔画像の横幅（16の倍数と仮定）
    # N: 出力の特徴ベクトルの次元数
    # K: クラスラベルの種類数
    def __init__(self, C, H, W, N, K):
        super(FaceEncoder2, self).__init__()

        # 畳込み層1
        # カーネルサイズ3，ストライド幅1，パディング1の設定なので，これを通しても特徴マップの縦幅・横幅は変化しない
        self.conv1 = nn.Conv2d(in_channels=C, out_channels=16, kernel_size=3, stride=1, padding=1)

        # 畳込み層2～5
        # カーネルサイズ4，ストライド幅2，パディング1の設定なので，これらを通すことにより特徴マップの縦幅・横幅がそれぞれ 1/2 になる
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)

        # バッチ正規化層
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.bn5 = nn.BatchNorm2d(num_features=128)

        # 平坦化
        self.flat = nn.Flatten()

        # 画像情報を処理する全結合層
        # 畳込み層2～5を通すことにより特徴マップの縦幅・横幅は都合 1/16 になっている．
        # したがって，入力側のパーセプトロン数は 128*(H/16)*(W/16) = H*W/2
        self.fc_img = nn.Linear(in_features=H*W//2, out_features=1024) # 画像情報は最終的に1024次元に

        # ラベル情報を処理する全結合層
        self.fc_lab1 = nn.Linear(in_features=K, out_features=256)
        self.fc_lab2 = nn.Linear(in_features=256, out_features=1024) # ラベル情報も1024次元に

        # 画像・ラベル情報の結合後に用いる全結合層
        self.fc_mu = nn.Linear(in_features=2048, out_features=N) # 1024次元になった画像情報とラベル情報を結合するので，トータルで2048次元
        self.fc_lnvar = nn.Linear(in_features=2048, out_features=N)

    def forward(self, x, y):
        # 画像情報 x を処理
        h = F.leaky_relu(self.bn1(self.conv1(x)))
        h = F.leaky_relu(self.bn2(self.conv2(h)))
        h = F.leaky_relu(self.bn3(self.conv3(h)))
        h = F.leaky_relu(self.bn4(self.conv4(h)))
        h = F.leaky_relu(self.bn5(self.conv5(h)))
        h = self.flat(h)
        hx = torch.tanh(self.fc_img(h))

        # ラベル情報 y を処理
        h = F.leaky_relu(self.fc_lab1(y))
        hy = torch.tanh(self.fc_lab2(h))

        # 画像情報とラベル情報を結合
        h = torch.cat((hx, hy), dim=1)

        # 特徴分布の平均・分散を計算し，特徴ベクトルを一つサンプリング
        mu = self.fc_mu(h)
        lnvar = self.fc_lnvar(h)
        eps = torch.randn_like(mu) # mu と同じサイズの標準正規乱数を生成
        z = mu + eps * torch.exp(0.5 * lnvar)
        return z, mu, lnvar


# N 次元の特徴ベクトルから顔画像を生成するニューラルネットワーク（CVAE版）
# CVAEのデコーダ部分のサンプル
class FaceDecoder2(nn.Module):

    # C: 入力顔画像のチャンネル数（1または3と仮定）
    # H: 入力顔画像の縦幅（16の倍数と仮定）
    # W: 入力顔画像の横幅（16の倍数と仮定）
    # N: 出力の特徴ベクトルの次元数
    # K: クラスラベルの種類数
    def __init__(self, C, H, W, N, K):
        super(FaceDecoder2, self).__init__()
        self.W = W
        self.H = H

        # ラベル情報を処理する全結合層
        self.fc_lab1 = nn.Linear(in_features=K, out_features=256)
        self.fc_lab2 = nn.Linear(in_features=256, out_features=1024) # ラベル情報は最終的に1024次元に

        # 特徴ベクトルを処理する全結合層
        self.fc_feat = nn.Linear(in_features=N, out_features=1024) # 特徴ベクトルも1024次元に

        # ラベル情報と特徴ベクトルの統合後に用いる全結合層
        self.fc_all = nn.Linear(in_features=2048, out_features=H*W//2) # 1024次元になったラベル情報と特徴ベクトルを結合するので，トータルで2048次元

        # 逆畳込み層1～4
        # カーネルサイズ，ストライド幅，パディングは FaceEncoder2 の畳込み層2～5と真逆に設定
        self.deconv4 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)

        # Residual Block 1〜4
        # checker board artifact の補正を狙いとして逆畳み込み層の直後に入れる
        self.rb4 = ResBlock(channels=128, kernel_size=3, stride=1, padding=1)
        self.rb3 = ResBlock(channels=64, kernel_size=3, stride=1, padding=1)
        self.rb2 = ResBlock(channels=32, kernel_size=3, stride=1, padding=1)
        self.rb1 = ResBlock(channels=16, kernel_size=3, stride=1, padding=1)

        # バッチ正規化層
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn1 = nn.BatchNorm2d(num_features=16)

        # 畳込み層
        self.conv = nn.Conv2d(in_channels=16, out_channels=C, kernel_size=3, stride=1, padding=1)

    def forward(self, z, y):
        # 特徴ベクトル z を処理
        hz = torch.tanh(self.fc_feat(z))

        # ラベル情報 y を処理
        h = F.leaky_relu(self.fc_lab1(y))
        hy = torch.tanh(self.fc_lab2(h))

        # 特徴ベクトルとラベル情報を結合
        h = torch.cat((hz, hy), dim=1)

        # 画像を生成
        h = F.leaky_relu(self.fc_all(h))
        h = torch.reshape(h, (len(h), 128, self.H//16, self.W//16)) # 一列に並んだパーセプトロンを 64*(H/8)*(W/8) の特徴マップに並べ直す
        h = F.leaky_relu(self.bn4(self.deconv4(h)))
        h = self.rb4(h)
        h = F.leaky_relu(self.bn3(self.deconv3(h)))
        h = self.rb3(h)
        h = F.leaky_relu(self.bn2(self.deconv2(h)))
        h = self.rb2(h)
        h = F.leaky_relu(self.bn1(self.deconv1(h)))
        h = self.rb1(h)
        y = torch.sigmoid(self.conv(h))
        return y


# 顔画像生成ニューラルネットワーク
# GAN生成器（ジェネレータ）のサンプル
class Generator(nn.Module):

    # C: 入力顔画像のチャンネル数（1または3と仮定）
    # H: 入力顔画像の縦幅（16の倍数と仮定）
    # W: 入力顔画像の横幅（16の倍数と仮定）
    # N: 出力の特徴ベクトルの次元数
    def __init__(self, C, H, W, N):
        super(Generator, self).__init__()
        self.W = W
        self.H = H

        # 全結合層
        # 逆畳込み層に通す前の時点で，特徴マップのサイズが 512 * (H/16) * (W/16) = 2*W*H となるようにしておく
        self.fc = nn.Linear(in_features=N, out_features=2*W*H)

        # 逆畳込み層1～4
        # カーネルサイズ4，ストライド幅2，パディング1の設定なので，これらを通すことにより特徴マップの縦幅・横幅がそれぞれ 2 倍になる
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)

        # バッチ正規化層
        self.bn1d = nn.BatchNorm2d(num_features=256)
        self.bn2d = nn.BatchNorm2d(num_features=128)
        self.bn3d = nn.BatchNorm2d(num_features=64)
        self.bn4d = nn.BatchNorm2d(num_features=32)
        self.bn1c = nn.BatchNorm2d(num_features=256)
        self.bn2c = nn.BatchNorm2d(num_features=128)
        self.bn3c = nn.BatchNorm2d(num_features=64)

        # 畳み込み層1〜5
        # カーネルサイズ3，ストライド幅1，パディング1の設定なので，これらを通しても特徴マップの縦幅・横幅は変化しない
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=C, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = F.leaky_relu(self.fc(z))
        h = torch.reshape(h, (len(h), 512, self.H//16, self.W//16)) # 一列に並んだパーセプトロンを 512*(H/16)*(W/16) の特徴マップに並べ直す
        h = F.leaky_relu(self.bn1d(self.deconv1(h)))
        h = F.leaky_relu(self.bn1c(self.conv1(h)))
        h = F.leaky_relu(self.bn2d(self.deconv2(h)))
        h = F.leaky_relu(self.bn2c(self.conv2(h)))
        h = F.leaky_relu(self.bn3d(self.deconv3(h)))
        h = F.leaky_relu(self.bn3c(self.conv3(h)))
        h = F.leaky_relu(self.bn4d(self.deconv4(h)))
        y = torch.sigmoid(self.conv4(h))
        return y


# 顔画像が Real か Fake を判定するニューラルネットワーク
# GAN識別器（ディスクリミネータ）のサンプル
class Discriminator(nn.Module):

    # C: 入力顔画像のチャンネル数（1または3と仮定）
    # H: 入力顔画像の縦幅（16の倍数と仮定）
    # W: 入力顔画像の横幅（16の倍数と仮定）
    # N: 出力の特徴ベクトルの次元数
    def __init__(self, C, H, W, N):
        super(Discriminator, self).__init__()

        # 畳込み層1～4
        # カーネルサイズ4，ストライド幅2，パディング1の設定なので，これらを通すことにより特徴マップの縦幅・横幅がそれぞれ 1/2 になる
        self.conv1 = nn.Conv2d(in_channels=C, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)

        # バッチ正規化層
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.bn4 = nn.BatchNorm2d(num_features=128)

        # ドロプアウト層
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)
        self.drop4 = nn.Dropout(p=0.5)

        # 平坦化
        self.flat = nn.Flatten()

        # 全結合層1
        # 畳込み層1～4を通すことにより特徴マップの縦幅・横幅は都合 1/16 になっており，
        # その後，さらに self.conv4 を通してから全結合層を適用する予定なので，入力側のパーセプトロン数は 128*(H/16)*(W/16) = H*W/2
        self.fc1 = nn.Linear(in_features=H*W//2, out_features=256)

        # 全結合層2
        self.fc2 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        h = F.leaky_relu(self.drop1(self.bn1(self.conv1(x))))
        h = F.leaky_relu(self.drop2(self.bn2(self.conv2(h))))
        h = F.leaky_relu(self.drop3(self.bn3(self.conv3(h))))
        h = F.leaky_relu(self.drop4(self.bn4(self.conv4(h))))
        h = self.flat(h)
        h = F.leaky_relu(self.fc1(h))
        z = torch.sigmoid(self.fc2(h))
        return z


# 顔画像生成ニューラルネットワーク
# CGAN生成器（ジェネレータ）のサンプル
class Generator2(nn.Module):

    # C: 入力顔画像のチャンネル数（1または3と仮定）
    # H: 入力顔画像の縦幅（16の倍数と仮定）
    # W: 入力顔画像の横幅（16の倍数と仮定）
    # N: 出力の特徴ベクトルの次元数
    # K: クラスラベルの種類数
    def __init__(self, C, H, W, N, K):
        super(Generator2, self).__init__()
        self.W = W
        self.H = H

        # ラベル情報を処理する全結合層
        self.fc_lab = nn.Linear(in_features=K, out_features=128) # ラベル情報を128次元に

        # 特徴ベクトルを処理する全結合層
        self.fc_feat = nn.Linear(in_features=N, out_features=128) # 特徴ベクトルも128次元に拡張

        # 全結合層
        # 逆畳込み層に通す前の時点で，特徴マップのサイズが 512 * (H/16) * (W/16) = 2*W*H となるようにしておく
        self.fc_all = nn.Linear(in_features=256, out_features=2*W*H)

        # 逆畳込み層1～4
        # カーネルサイズ4，ストライド幅2，パディング1の設定なので，これらを通すことにより特徴マップの縦幅・横幅がそれぞれ 2 倍になる
        self.deconv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1)

        # バッチ正規化層
        self.bn1d = nn.BatchNorm2d(num_features=256)
        self.bn2d = nn.BatchNorm2d(num_features=128)
        self.bn3d = nn.BatchNorm2d(num_features=64)
        self.bn4d = nn.BatchNorm2d(num_features=32)
        self.bn1c = nn.BatchNorm2d(num_features=256)
        self.bn2c = nn.BatchNorm2d(num_features=128)
        self.bn3c = nn.BatchNorm2d(num_features=64)

        # 畳み込み層1〜5
        # カーネルサイズ3，ストライド幅1，パディング1の設定なので，これらを通しても特徴マップの縦幅・横幅は変化しない
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=C, kernel_size=3, stride=1, padding=1)

    def forward(self, z, y):
        hy = torch.tanh(self.fc_lab(y))
        hz = torch.tanh(self.fc_feat(z))
        h = torch.cat((hz, hy), dim=1) # 乱数ベクトルとラベル情報を連結
        h = F.leaky_relu(self.fc_all(h))
        h = torch.reshape(h, (len(h), 512, self.H//16, self.W//16)) # 一列に並んだパーセプトロンを 512*(H/16)*(W/16) の特徴マップに並べ直す
        h = F.leaky_relu(self.bn1d(self.deconv1(h)))
        h = F.leaky_relu(self.bn1c(self.conv1(h)))
        h = F.leaky_relu(self.bn2d(self.deconv2(h)))
        h = F.leaky_relu(self.bn2c(self.conv2(h)))
        h = F.leaky_relu(self.bn3d(self.deconv3(h)))
        h = F.leaky_relu(self.bn3c(self.conv3(h)))
        h = F.leaky_relu(self.bn4d(self.deconv4(h)))
        y = torch.sigmoid(self.conv4(h))
        return y


# 顔画像が Real か Fake を判定するニューラルネットワーク
# CGAN識別器（ディスクリミネータ）のサンプル
class Discriminator2(nn.Module):

    # C: 入力顔画像のチャンネル数（1または3と仮定）
    # H: 入力顔画像の縦幅（16の倍数と仮定）
    # W: 入力顔画像の横幅（16の倍数と仮定）
    # N: 出力の特徴ベクトルの次元数
    # K: クラスラベルの種類数
    def __init__(self, C, H, W, N, K):
        super(Discriminator2, self).__init__()

        # 畳込み層1～4
        # カーネルサイズ4，ストライド幅2，パディング1の設定なので，これらを通すことにより特徴マップの縦幅・横幅がそれぞれ 1/2 になる
        self.conv1img = nn.Conv2d(in_channels=C, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.conv1lab = nn.Conv2d(in_channels=K, out_channels=8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)

        # バッチ正規化層
        self.bn1img = nn.BatchNorm2d(num_features=8)
        self.bn1lab = nn.BatchNorm2d(num_features=8)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.bn4 = nn.BatchNorm2d(num_features=128)

        # ドロプアウト層
        #self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)
        self.drop4 = nn.Dropout(p=0.5)

        # 平坦化
        self.flat = nn.Flatten()

        # 全結合層1
        # 畳込み層1～4を通すことにより特徴マップの縦幅・横幅は都合 1/16 になっており，
        # その後，さらに self.conv4 を通してから全結合層を適用する予定なので，入力側のパーセプトロン数は 128*(H/16)*(W/16) = H*W/2
        self.fc1 = nn.Linear(in_features=H*W//2, out_features=256)

        # 全結合層2
        self.fc2 = nn.Linear(in_features=256, out_features=1)

        '''
        # ラベル情報を処理する全結合層
        self.fc_lab = nn.Linear(in_features=K, out_features=128) # ラベル情報は128次元に

        # 画像情報を処理する全結合層
        # 畳込み層1～4を通すことにより特徴マップの縦幅・横幅は都合 1/16 になっており，
        # その後，さらに self.conv4 を通してから全結合層を適用する予定なので，入力側のパーセプトロン数は 128*(H/16)*(W/16) = H*W/2
        self.fc_img = nn.Linear(in_features=H*W//2, out_features=128) # 画像情報も128次元に

        # 全結合層
        self.fc_all1 = nn.Linear(in_features=256, out_features=64) # 128次元になった画像情報とラベル情報を結合するので，トータルで256次元
        self.fc_all2 = nn.Linear(in_features=64, out_features=1)
        '''

    def forward(self, x, y):
        hx = F.leaky_relu(self.bn1img(self.conv1img(x)))
        hy = F.leaky_relu(self.bn1lab(self.conv1lab(y)))
        h = torch.cat((hx, hy), dim=1) # 画像情報とラベル情報を結合
        h = F.leaky_relu(self.drop2(self.bn2(self.conv2(h))))
        h = F.leaky_relu(self.drop3(self.bn3(self.conv3(h))))
        h = F.leaky_relu(self.drop4(self.bn4(self.conv4(h))))
        h = self.flat(h)
        h = F.leaky_relu(self.fc1(h))
        z = torch.sigmoid(self.fc2(h))
        '''
        hx = torch.tanh(self.fc_img(h))
        hy = torch.tanh(self.fc_lab(y))
        h = torch.cat((hx, hy), dim=1) # 画像情報とラベル情報を結合
        h = F.leaky_relu(self.fc_all1(h))
        z = torch.sigmoid(self.fc_all2(h))
        '''
        return z
