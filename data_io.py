import os
import numpy as np
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.utils.data import Dataset


# CSVファイルをデータセットとして扱うためのクラス
class CSVDataset(Dataset):

    # コンストラクタ，CSVDatasetクラスのインスタンスを作成した時点で呼ばれる
    # この時点でCSVファイルの情報を全て読み込み，メンバ変数に保存するようにしておく
    def __init__(self, filename, input, output=None, alloc=None):
        super().__init__()

        # pandasを用いてCSVファイルを読み込む
        self.df = pd.read_csv(filename)

        # 引数 input で指定された項目を入力として扱う => torch.Tensor型のメンバ変数 X に保存
        self.X = torch.tensor(self.df[input].values, dtype=torch.float32, device='cpu')

        # 引数 output で指定された項目を出力として扱う => torch.Tensor型メンバ変数 Y に保存
        if output is None:
            self.Y = None
        else:
            if alloc is None:
                self.Y = torch.tensor(self.df[output].values, dtype=torch.long, device='cpu')
            else:
                self.Y = torch.tensor(self.df[output].replace(alloc).values, dtype=torch.long, device='cpu')

        # データセットサイズ（データ数）を記憶しておく => int型のメンバ変数 len に保存
        self.len = len(self.X)

    # データセットサイズを返却する関数
    def __len__(self):
        return self.len

    # index 番目のデータを返却する関数
    # データローダは，この関数を必要な回数だけ呼び出して，自動的にミニバッチを作成してくれる
    def __getitem__(self, index):
        x = self.X[index] # 入力値
        if self.Y is None:
            return x
        else:
            y = self.Y[index] # 出力値（正解値）
            return x, y


# 画像データセットを扱うためのクラス
# 画像ファイル名の一覧をクラスラベルとともに記載したリストファイル（CSV）が必要
class ImageDataset(Dataset):

    # コンストラクタ
    # この時点で画像ファイル名の一覧は全て読み込むが，画像そのものは読み込まない（メモリが足りなくなるため）
    def __init__(self, filename, dirname, input, output=None, alloc=None):
        super().__init__()

        # pandasを用いてCSVファイル（画像ファイル名リスト）を読み込む
        self.df = pd.read_csv(filename)

        # 引数 input で指定された項目を入力ファイル名として扱う => list型のメンバ変数 X に保存
        self.dirname = dirname
        self.X = self.df[input].to_list()

        # 引数 output で指定された項目を出力として扱う => torch.Tensor型メンバ変数 Y に保存
        if output is None:
            self.Y = None
        elif type(output) is list:
            n = len(output)
            Y = []
            if alloc is None:
                for i in range(n):
                    Y.append(self.df[output[i]].values)
            else:
                for i in range(n):
                    Y.append(self.df[output[i]].replace(alloc[i]).values)
            self.Y = torch.tensor(np.asarray(Y).T, dtype=torch.float32, device='cpu')
        else:
            if alloc is None:
                self.Y = torch.tensor(self.df[output].values, dtype=torch.long, device='cpu')
            else:
                self.Y = torch.tensor(self.df[output].replace(alloc).values, dtype=torch.long, device='cpu')

        # データセットサイズ（データ数）を記憶しておく => int型のメンバ変数 len に保存
        self.len = len(self.X)

    # データセットサイズを返却する関数
    def __len__(self):
        return self.len

    # index 番目のデータを返却する関数
    # データローダは，この関数を必要な回数だけ呼び出して，自動的にミニバッチを作成してくれる
    def __getitem__(self, index):
        # 実際に画像ファイルを読み込み，画素値を 0～1 に正規化する（元々が 0~255 なので，255 で割る）
        x = torchvision.io.read_image(os.path.join(self.dirname, self.X[index])) / 255 # 画像そのものが入力値
        if self.Y is None:
            return x
        else:
            y = self.Y[index] # 出力値（正解値）
            return x, y


# 画像表示用関数
# 学習処理そのものとは無関係
def show_image(data, title='no title'):
    img = np.asarray(data * 255, dtype=np.uint8)[0].transpose(1, 2, 0)
    plt.axis('off')
    plt.title(title)
    plt.imshow(img, cmap=cm.gray, interpolation='nearest')
    plt.pause(2)
    plt.close()


# 複数枚の画像を表示する関数
# 経過確認用，学習処理そのものとは無関係
#   - n_data_max: 表示する画像の枚数
#   - n_data_per_row: 1行あたりの表示枚数
#   - save_fig: 表示結果をファイルにも保存するか否か
def show_images(data, title='no title', n_data_max=100, n_data_per_row=10, save_fig=False):
    data = np.asarray(data * 255, dtype=np.uint8)
    n_data_total = min(data.shape[0], n_data_max) # 保存するデータの総数
    n_rows = n_data_total // n_data_per_row # 保存先画像においてデータを何行に分けて表示するか
    if n_data_total % n_data_per_row != 0:
        n_rows += 1
    plt.figure(title, figsize=(n_data_per_row, n_rows))
    for i in range(0, n_data_total):
        plt.subplot(n_rows, n_data_per_row, i+1)
        plt.axis('off')
        plt.imshow(data[i].transpose(1, 2, 0), cmap=cm.gray, interpolation='nearest')
    plt.pause(2)
    if save_fig:
        plt.savefig(title + '.png', bbox_inches='tight')
    plt.close()
