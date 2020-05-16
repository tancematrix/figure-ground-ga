# coding:utf-8
"""
目標となる文字列画像を読み込んで、それに近づけるように円画像を配置していく。
"""
from genetic_algorithm import *
import argparse
import numpy

def decide_genom_length(target: np.ndarray):
    """
    目標となる画像のサイズから、配置すべき円の数を概算する。
    """
    height, width = target.shape
    # 円の半径は、区間[0, 短辺の1/2]の一様分布からサンプルするので、期待値は短辺の1/4である。
    expected_r = min(height, width) / 4
    expected_circle_size = expected_r ** 2 / np.pi
    # 画像サイズを平均の円の面積で割って、（厳密に詰め込まれるわけではないから）ちょっとかさまし
    return int(height * width / expected_circle_size * 1.4)

if __name__ == "__main__":
    """
    定数
    GENERATION_GAP: 世代交代で淘汰される個体の割合
    GENERATION_SIZE: 一世代の個体数
    MAX_ITER: 最大の世代交代数（ループの数）
    PM: 突然変異確率（各遺伝子がこの確率で突然変異する）
    """
    GENERATION_GAP = 0.2
    GENERATION_SIZE = 100
    PM = 0.05

    parser = argparse.ArgumentParser()
    parser.add_argument("--target", action="store", type=str, required=True, help="Path to the image of letters, which we imitate. Use letter2image.py beforehand.")
    parser.add_argument("--font_path", action="store", type=str, default='/System/Library/Fonts/SFCompactDisplay-Light.otf', help="Path to your font. Both of ttf and otf work.")
    parser.add_argument("--iter_num", action="store", type=int, default=5000, help="Number of generation iterations")    
    args = parser.parse_args()

    # 画像を読み込んでグレースケールに変換
    img = PIL.Image.open(args.target)
    target = np.array(img.convert('L'))
    height, width = target.shape
    circle_num = decide_genom_length(target)

    g = Generation(generation_size=GENERATION_SIZE, genom_length=circle_num, genom_limits=[height, width, width // 2])
    g.set_pm(PM)
    g.print_info()
    for i in range(args.iter_num):
        g.evaluate(255-target)
        if i % 10 == 0:
            print(f"generation {i}")
            mi, ma, ave = g.summary()
            if mi < 100:
                print("score achieved, break..")
                break
        if i % 100 == 0:
            print("saving 途中経過...")
            g.show_all(255-target, savepath=f"results/generation_{i}.png")
        g.select(int(GENERATION_SIZE * (1 - GENERATION_GAP)))
        g.breed()
    best_genom, best_score = g.chief()
    best_pheno = Phenotype(g, shape=target.shape)
    best_pheno.save("results/best.jpg")
    pickle.dump(g, open("results/last_generation.pkl", "wb"))
