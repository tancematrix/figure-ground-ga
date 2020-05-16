# coding:utf-8
"""
目標となる文字列画像を読み込んで、それに近づけるように円画像を配置していく。
"""
from genetic_algorithm import *
import PIL.Image
import argparse
import numpy

def decide_genom_length(target: np.ndarray):
    """
    目標となる画像のサイズから、配置すべき円の数を概算する。
    circle_numを変えながらGenerationを初期化してみて、被覆率が0.65を超えたあたりのcircle_numを採用
    """
    print("Determining the number of circles to locate....")
    circle_num = 5
    height, width = target.shape
    r = min(width,height) // 2
    while(True):
        g = Generation(generation_size=100, genom_length=circle_num, genom_limits=[height, width, r])
        coverage = np.mean([np.sum(Phenotype(ge, shape=target.shape).as_image()==0) / (height*width) for ge in g.genom_list])
        print(f"circle_num={circle_num}, coverage={coverage}")
        if coverage > 0.9:
            break
        circle_num += 5
    print("Done")
    return circle_num

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
    parser.add_argument("--iter_num", action="store", type=int, default=5000, help="Number of generation iterations")    
    parser.add_argument("--result_dir", action="store", type=str, default="result", help="Directory to save result files.")    
    args = parser.parse_args()

    # 画像を読み込んでグレースケールに変換
    img = PIL.Image.open(args.target)
    target = np.array(img.convert('L'))
    height, width = target.shape
    circle_num = decide_genom_length(target)

    g = Generation(generation_size=GENERATION_SIZE, genom_length=circle_num, genom_limits=[height, width, min(height, width) // 2])
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
            g.show_all(255-target, savepath=args.result_dir+f"/generation_{i}.png")
        g.select(int(GENERATION_SIZE * (1 - GENERATION_GAP)))
        g.breed()
    g.show_all(255-target, savepath=args.result_dir+f"/generation_{args.iter_num-1}.png")
    best_genom, best_score = g.chief()
    best_pheno = Phenotype(best_genom, shape=target.shape)
    best_pheno.save(args.result_dir + "/best.jpg")
    pickle.dump(g, open(args.result_dir + "/last_generation.pkl", "wb"))
