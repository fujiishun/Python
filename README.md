このアプリケーションはPythonを用いて開発された、画像認識アプリケーションです。

開発環境
Python 3.8

ライブラリ
Pillow
streamlit
torch
torchvision
transformers
cloudpickle

機械学習モデルについて
本アプリケーションでは、.pth形式の機械学習モデルが搭載されています。このモデルは(Pytorch、Tensorflow、Scikit-learn、その他に基づいて開発が行われた)をベースにしたもので、画像認識のために訓練されています。

セットアップ
本アプリケーションは以下の手順でセットアップできます。
$ pip install -r requirements.txt

アプリケーションの起動
以下を実行します。
$ streamlit run app.py