# OCRによる表データのCSV変換

## 説明

- csvへの出力結果は`src/output.csv`にあります。
- ソースコードは`src/`にあります。
- 画像の処理の過程は`src/img/output`にあります。

## プロジェクト設定

- プロジェクトは`poetry`で管理しているので、`poetry`をインストールしてください。
- `poetry`のインストール方法は[こちら](https://python-poetry.org/docs/#installation)を参照してください。
- `poetry`をインストールしたら、以下のコマンドを実行してください。
- なお、以下で動作確認をしています。（pyproject.tomlを参照）
  - Python 3.11.5
  - opencv-python 4.8.1.78
  - easyocr 1.7.1

 ```bash
 cd table_ocr-main # プロジェクトのルートディレクトリに移動
 poetry config virtualenvs.in-project true # 仮想環境をプロジェクト内に作成
 poetry shell # 仮想環境を有効化
 poetry install # ライブラリをインストール
 ```

## 実行方法

以下のコマンドで実行できます。（初回実行に時間がかかる場合があります。一度終了して再実行すると完了する場合があります。）

```bash
cd src
python main.py
```

実行すると、`src/img/table.png`の画像から読み取った表データから作成した`src/output.csv`が作成されます。

また、以下のようにしてデバッグモードで実行できます。
デバッグモードでは、画像の処理の過程を`src/img/output`に保存します。

```bash
python main.py True
```
