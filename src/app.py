# 必要なモジュールのインポート
import torch
from .animal import transform, Net # animal.py から前処理とネットワークの定義を読み込み
from flask import Flask, request, render_template, redirect 
import io
from PIL import Image
import base64

# 学習済みモデルをもとに推論する
def predict(img):
    # ネットワークの準備
    net = Net().cpu().eval()
    # # 学習済みモデルの重み（dog_cat.pt）を読み込み
    net.load_state_dict(torch.load('./src/dog_cat.pt', map_location=torch.device('cpu')))
    #　データの前処理
    img = transform(img)
    img =img.unsqueeze(0) # 1次元増やす
    #　推論
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y

#　推論したラベルから犬か猫かを返す関数
def getName(label):
    if label==0:
        return '猫'
    elif label==1:
        return '犬'

# Flask のインスタンスを作成
app = Flask(__name__)
app.secret_key = 'secret_key'

# アップロードされる拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif', 'jpeg'])

#　拡張子が適切かどうかをチェック
def allwed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# アスペクト比を固定して、幅が指定した値になるようリサイズする。
def scale_to_width(img, width):
    height = round(img.height * width / img.width)
    return img.resize((width, height))

# URL にアクセスがあった場合の挙動の設定
@app.route('/', methods = ['GET', 'POST'])
def predicts():
    # リクエストがポストかどうかの判別
    if request.method == 'POST':
        # ファイルがなかった場合の処理
        if 'filename' not in request.files:
            return redirect(request.url)
        # データの取り出し
        file = request.files['filename']
        # ファイルのチェック
        if file and allwed_file(file.filename):

            #　画像ファイルに対する処理

            #　画像書き込み用バッファを確保
            buf = io.BytesIO()
            image = Image.open(file).convert('RGB')

            # リサイズ
            if image.width > 512:
                image = scale_to_width(image, 512)

            #　画像データをバッファに書き込む
            image.save(buf, 'png')
            #　バイナリデータを base64 でエンコードして utf-8 でデコード
            base64_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            #　HTML 側の src  の記述に合わせるために付帯情報付与する
            base64_data = 'data:image/png;base64,{}'.format(base64_str)

            # 入力された画像に対して推論
            pred = predict(image)
            animalName_ = getName(pred)
            return render_template('result.html', animalName=animalName_, image=base64_data)
        else :
            return redirect(request.url)

    # GET メソッドの定義
    elif request.method == 'GET':
        return render_template('index.html')


# アプリケーションの実行の定義
if __name__ == '__main__':
    app.run(debug=True)
