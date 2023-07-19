# tf-torch-sample
tensorflow, pytorchで書かれた.py, .ipynbファイルによるGPU動作確認手順
(注: 一般向けの内容ではありません)

# 実行までの手順

## 準備(サンプルコードのclone、仮想環境作成)
```bash
# 所望のディレクトリにcloneする
cd /share/(自分の名前ディレクトリ)
git clone https://github.com/rtmk0525/tf-torch-sample.git
cd tf-torch-sample
# 以下、仮想環境の構築
# anacondaのpathを通す(?はユーザ番号)
. /home/user?/set-anaconda.sh
# 仮想環境作成(それぞれpytorch用、tensorflow用)
conda create -n torch-sample python=3.11
conda create -n tf-sample python=3.11
```

## pytorchの動作確認
```bash
# pytorch用の仮想環境有効化
. activate torch-sample
# pythonライブラリをいくつかインストール
pip install torch torchvision ipykernel
# jupyter notebookに仮想環境`torch-sample`を追加
# 参考: https://qiita.com/smiler5617/items/e0d9b3034d79457cc253
ipython kernel install --user --name=torch-sample
```
### .pyの実行手順
```bash
python pytorch.py
```
### .ipynbの実行手順
```bash
# jupyter notebook起動
jupyter notebook
```
以下、`pytorch.ipynb`を開き、カーネル→カーネルの変更→`torch-sample`を選択し、実行する。

## tensorflowの動作確認
```bash
# 必要なら現在の仮想環境を抜ける
. deactivate
# tensorflowo用の仮想環境有効化
. activate tf-sample
# pythonライブラリをいくつかインストール
pip install tensorflow ipykernel
# jupyter notebookに仮想環境`tf-sample`を追加
# 参考: https://qiita.com/smiler5617/items/e0d9b3034d79457cc253
ipython kernel install --user --name=tf-sample
# 重要: GPUを利用できるようにする
# 参考: https://www.tensorflow.org/install/pip?hl=ja#linux_1
conda install -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163
# システムパス設定の自動化バージョン（仮想環境を起動するたびにシステムパスが自動的に構成される）
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# 仮想環境に入り直す
. deactivate
. activate tf-sample
```
### .pyの実行手順
```bash
python tf.py
```
### .ipynbの実行手順
```bash
# jupyter notebook起動
jupyter notebook
```
以下、`tf.ipynb`を開き、カーネル→カーネルの変更→`tf-sample`を選択し、実行する。

# 当該環境での結果
`.py`の結果は省略

## `pytorch.ipynb`の結果
```bash
device=device(type='cuda', index=1)
(後略)
```
GPUを利用できている。

## `tf.ipynb`の結果
```bash
(前略)
logical_gpus=[LogicalDevice(name='/device:GPU:0', device_type='GPU')]
(後略)
```
GPUを利用できている。
