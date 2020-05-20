# figure-ground-ga
Using genetic algorithm, locate circles so that the background looks like a letter.

背景が文字に見えるように、円を配置します。遺伝的アルゴリズムを用いています。
「2m離れると読める」と話題になった[岐阜新聞の広告の手法](https://news.yahoo.co.jp/articles/6a90515f951be8184e68b49677356d954548e60c)を自動化しようという試みです。

文字の画像を入力として、次のような出力をします。

# 実行方法
## requirements
- python3

    - 3.7.3と3.6.8においては動くことを確認しています
- packages
    - Pillow(文字画像の生成と画像出力のために)、numpy, scipy
    - 詳細はrequirments.txtを見てください。
## 文字画像の作成

    python letter2image.py --font_path FONT_PATH --output_file OUTPUT_FILE

「文字を入力してください」とプロンプトが出るので、何か文字を入力してください。1文字程度でないと難しいかもしれません。

- FONT_PATH: ttf/otfフォントへのpath
    - macなら"/System/Library/Fonts/"以下、windowsなら"C:/Windows/Fonts/"以下に色々あるかと思います。
- OUTPUT_FILE: 保存先の画像を指定できます。指定しなければ"./letters.png"に保存されます。

## 遺伝的アルゴリズムによる円の配置

    python main.py --target TARGET_IMAGE --iter_num ITER_NUM --circle_num CIRCLE_NUM --result_dir RESULT_DIR

- TARGET_IMAGE: 目標とする文字画像です。letters2image.pyでファイル名を指定していなければletters.pngです。
- ITER_NUM: 遺伝的アルゴリズムの世代交代の回数（上限）です。後述しますが、世代間最小ギャップ（MGG）モデルを採用しているため、一回の世代交代で入れ替わる個体は2つだけです。このため、たくさん世代交代を回さないと結果が出ません。10000くらいは回して見てください。ただし時間はかかります。
- CIRCLE_NUM: 配置する円の数です。指定しなければ適当に計算しますが、あくまで適当です。一文字につき、文字の複雑さに応じて10~30くらいかなと思います。
    - この円の数が結構クオリティに効いてくるので、それも含めてパラメータを獲得できるようにしたいです。
- RESULT_DIR: 適当なディレクトリを作成して、それを指定してください。そこにファイルが出力されます。

# 出力
- generation_X.png: X世代目の結果です。Xは500刻みです。
- best.png: 最終世代で最もスコアがよかった個体の画像です。
- last_generation.pkl: Generationオブジェクトをそのままpickleでdumpしたものです。ほとんどデバグにしか使えません。

# 中身について
## 遺伝的アルゴリズム
遺伝的アルゴリズムについては以下の資料を参考にしました。

https://www.slideshare.net/kzokm/genetic-algorithm-41617242

遺伝的アルゴリズムについての解説は省きますが、（上のスライドが非常にわかりやすくまとまっています）複数の戦略が考えられるような部分については以下のようなものを採用しました。

- エンコーディング: 実数値エンコーディング
    - ゲノムは1d-arrayとして表現するのが普通なのかもしれませんが、今回は[[円のx座標, 円のy座標, 円の半径] * 円の数]という実数値2d-arrayをゲノムとしました。
- 評価値: 後述
- 交差方法: 二点交差
- 突然変異: 置換（ランダムに選ばれた遺伝子を、一様乱数によって書き換える）のみ
- 世代交代: 世代間最小ギャップモデル
    1. ある世代からランダムに2個体を選んで交差し、子個体2体を得る（合計4個体）
    1. 合計4個体のうち、最も評価値の良い1個体を生存個体_1として選ぶ。
    1. 残った3個体のうちから、ルーレット選択によって確率的に1個体を選び、生存個体_2とする。
    1. 2体の生存個体によって親個体を置換したもの（ただし、親自身も生存個体となりうる）をを次世代とする。

## ゲノムの評価方法
「離れてみると文字に見える」かどうかを評価したいわけですが、以下のようにして再現しました。

1. target文字画像、生成画像共にガウシアンぼかしを施し、短辺が20pixとなるようにダウンサンプルします。
1. target文字画像、生成画像のl2ノルムを評価値とします。（小さい方が良い評価）

対応する箇所のコードはこうなっています。(self.morphというのが生成した画像に相当するnumpy arrayです)

    def evaluate(self, target:np.ndarray):
        ds_rate = min(self.morph.shape) // 20
        down_sampled_morph = scipy.ndimage.gaussian_filter(self.morph, sigma=ds_rate)[::ds_rate, ::ds_rate]
        down_sampled_target = scipy.ndimage.gaussian_filter(target, sigma=ds_rate)[::ds_rate, ::ds_rate]
        return np.linalg.norm(down_sampled_morph - down_sampled_target)

「遠くから見る」ということのモデルとしては妥当なものだと思います。画像のl2ノルムで距離を測るのは測るのはどうなんだという気もしますが、簡単のためにこうしました。

mnistなどで事前学習したCNNによる特徴量のcos距離...とかにするとさらに本格的だとは思いますが。

## 実装について
`genetic_algorithm.py`に遺伝的アルゴリズムのモジュールを実装しました。（pythonも遺伝的アルゴリズムも初心者なので、参考にするのはお勧めできません）

- Genom class
    - [円のx座標, 円のy座標, 円の半径] * 円の数 に相当する2d-arrayを格納することを主な目的としたクラス。
    - `mutate(pm)`メソッドで、突然変異確率=pmにおいて突然変異を施す。

- Phenotype class
    - genomの発現を管理するクラス。コンストラクタはGenomとshape:生成画像のshapeを引数にとる。genom:円の位置・サイズ情報を元に、画像に エンコードする。
    - `evaluate(target)`メソッドによってgenomの評価値を返す。targetはtarget文字画像。
        - 都合上、コード内では`255 - target`を渡している。
    - `save(path, title)`メソッドによって画像として保存する。

- Family class
    - 交差を司るクラス。コンストラクタは親となる2個体の`Genom`を引数にとる。
    - 使われているのは`mgg_change(target, pm)`メソッドのみ。世代間最小ギャップモデルに基づく世代交代における生存個体（2個体）を返す。

- Generation class
    - genomの集合である世代を管理するクラス。コンストラクタは`generation_size`:世代内の個体数, `genom_length`:ゲノムの長さ＝円の数, `genom_limits`:ゲノムを乱数で初期化する際の上限値=[画像の縦のpix数, 画像の横のpix数, 円の半径], `pm`:遺伝子ごとの突然変異確率 を引数にとる。
    - `evaluate(target)`メソッドで、Generationの中の各ゲノムの評価値を計算して保持する。
    - `mgg_change(target)`メソッドで世代間最小ギャップモデルに基づく世代交代を行う。
        - 基本的に上記二つのメソッドを繰り返すことで遺伝的アルゴリズムが実行される。
    - `summary()`:その世代の評価値のmin・max・averageをprintする, `print_info()`: 初期化時のパラメータを表示する, `show_all(target, path)`: その世代の結果を画像として保存する, などの事務的機能がある。

