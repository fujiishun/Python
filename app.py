import torch
import torchvision
from torchvision import transforms
from torchvision.models import mobilenetv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import cloudpickle

import streamlit as st
from PIL import Image

##CSSの定義
st.markdown(
"""
<style>
.header {
left: 0;
position: fixed;
width: 100%;
background-color: lightgray;
padding: 10px;
text-align: center;
font-size: 24px;
}
.sub-title {
text-align: left;
color: black;
font-size: 18px;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: lightgray;
color: black;
text-align: right;
padding: 10px;
}
</style>
""",
unsafe_allow_html=True,
)

##ヘッダ
st.markdown("<div class='header'>藤井の画像判定アプリ</div><br /><br /><br /><br />", unsafe_allow_html=True)

##左サイドバー
st.sidebar.markdown("## 操作説明")
st.sidebar.text("以下の手順で操作してください。")
st.sidebar.markdown("### 1.モデルをアップロードする")
st.sidebar.text("※モデル作成方法はnotionをご覧ください")
st.sidebar.text("※デフォルトでは藤井の自作モデルが適用されます")
st.sidebar.markdown("### 2.画像をアップロードする")
st.sidebar.text("※自身のモデルに合わせた画像を選んでください")
st.sidebar.text("※藤井の自作モデルの場合以下の判定が可能です")
st.sidebar.text("・バナナ")
st.sidebar.text("・りんご")
st.sidebar.text("・ぶどう")

##GPUの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([ # 検証データ用の画像の前処理
#transforms.RandomResizedCrop(size=(224,224),scale=(1.0,1.0),ratio=(1.0,1.0)), # アスペクト比を保って画像をリサイズ
transforms.Resize((224,224)),
transforms.ToTensor(),
transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

##モデルクラスの宣言
class Mobilenetv2(nn.Module):
    def __init__(self, pretrained_mob_model, class_num):
        super(Mobilenetv2, self).__init__()
        self.class_num = None
        self.vit = pretrained_mob_model #学習ずみモデル
        self.fc = nn.Linear(1000, class_num)
        self.categories = None

    def get_class_num(self):
        return self.class_num

    def forward(self, input_ids):
        states = self.vit(input_ids)
        states = self.fc(states)
        return states
upload_model = st.file_uploader('学習したAIモデルをアップロード',type=['pth'])

json_load = None

if upload_model is not None:
    try:
        net = torch.load(upload_model)
        features = net.categories
    except:
        st.write("モデルの読み込みに失敗しました、再度読み込み直してください")
        upload_model = None
else:
    upload_model = "model93.pth"
    net = torch.load(upload_model)
    features = net.categories

uploaded_file = st.file_uploader('判定する写真をアップロード', type=['jpg','png','jpeg'])
if uploaded_file is not None:

    img = Image.open(uploaded_file)

    data = torch.reshape(transform(img),(-1,3,224,224))

    net.eval()

    with torch.no_grad():
        out = net(data)
        predict = out.argmax(dim=1)
        #st.write(out)

    st.markdown('認識結果')

    if upload_model is not None:
        st.write(features[predict.detach().numpy()[0]])
    else:
        if json_load is not None:
            i = predict.detach().numpy()[0]
            st.write(json_load[str(i)])

    st.image(img)

##フッター
st.markdown("<div class='footer'>See you</div>", unsafe_allow_html=True)
