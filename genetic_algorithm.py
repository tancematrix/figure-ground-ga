# coding:utf-8
from matplotlib import pyplot as plt
import numpy as np
from typing import NamedTuple
import scipy.ndimage
import random
import pickle
import config
HEIGHT, WIDTH = 150, 90

# 補助的な関数
if config.numba_available:
    # numba使ってもそんなには早くならなかった
    from numba import jit
    @jit('b1[:,:](b1[:,:], u1, u1, u1)', nopython=True)
    def circle_mask(canvas, cx, cy, r):
        x = np.arange(canvas.shape[0]).reshape(-1, 1)
        y = np.arange(canvas.shape[1]).reshape(1, -1)
        r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
        mask = r2 > r ** 2
        return mask * canvas
else:
    def circle_mask(canvas, cx, cy, r):
        x = np.arange(canvas.shape[0]).reshape(-1, 1)
        y = np.arange(canvas.shape[1]).reshape(1, -1)
        r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
        mask = r2 > r ** 2
        return mask * canvas

class Genom:
    def __init__(self, genom_length, limits, random=True, chromosome=None):
        self.genom_length = np.ceil(np.random.normal(genom_length, scale=(0.01 * genom_length))) # 0 ~ genom_length * 2の間に存在する確率がだいたい99.7%
        self.genom_length = int(max(3, self.genom_length)) # 3以上ないと交差ができない
        self.gene_length = 3 # ハードコーディング
        self.chromosome = np.zeros([self.genom_length, self.gene_length], dtype=np.int)
        self.limits = limits
        self.evaluation = None
        if chromosome is not None:
            self.chromosome = chromosome
        elif random:
            self.randomize()

    def getlimits(self):
        return self.limits

    def set_evaluation(self, evaluation):
        self.evaluation = evaluation

    def randomize(self):
        # ハードコーディング
        self.chromosome[:,0] = np.random.randint(0, self.limits[0], self.genom_length)
        self.chromosome[:,1] = np.random.randint(0, self.limits[1], self.genom_length)
        self.chromosome[:,2] = np.random.randint(0, self.limits[2], self.genom_length) # 円の半径
    
    def mutate(self, pm):
        # Pm(変異率)が、「あるゲノムで突然変異が起こる確率」なのか、「ある遺伝子座で突然変異が起こる確率」なのか理解できていない。
        # 挿入・欠失は前者、置換は後者の解釈でpmを与えているため、一貫していない。
        # ただ、この確率がアルゴリズムの成否に強く関わる訳ではないと考えるので、一旦こういう実装にしてある。
        self.insert_delete(pm/3)
        self.perturbate(pm/3)
#        self.replace(pm/3)

    def replace(self, pm):
        whichwillbechanged = np.random.choice([True, False], self.chromosome.shape, p=[pm, 1-pm])
        self.chromosome[whichwillbechanged[:,0], 0] = np.random.randint(0, self.limits[0], np.sum(whichwillbechanged[:,0]))
        self.chromosome[whichwillbechanged[:,1], 1] = np.random.randint(0, self.limits[1], np.sum(whichwillbechanged[:,1]))
        self.chromosome[whichwillbechanged[:,2], 2] = np.random.randint(0, self.limits[2], np.sum(whichwillbechanged[:,2]))

    def perturbate(self, pm):
        whichwillbechanged = np.random.choice([True, False], self.chromosome.shape, p=[pm, 1-pm])
        self.chromosome[whichwillbechanged[:,0], 0] += np.floor(np.random.normal(0, scale=10, size=np.sum(whichwillbechanged[:,0]))).astype(np.int)
        self.chromosome[whichwillbechanged[:,1], 1] += np.floor(np.random.normal(0, scale=10, size=np.sum(whichwillbechanged[:,1]))).astype(np.int)
        self.chromosome[whichwillbechanged[:,2], 2] += np.floor(np.random.normal(0, scale=10, size=np.sum(whichwillbechanged[:,2]))).astype(np.int)
        self.chromosome[self.chromosome < 0] = 0 # 負にならないように

    def insert_delete(self, pm):
        operation = np.random.choice(["delete", "insert", "none"], 1, p=[pm/2, pm/2, 1-pm])[0]
        if operation == "none":
            pass
        elif operation == "delete":
            ind = np.random.randint(0, len(self.chromosome), 1)[0]
            self.chromosome = np.delete(self.chromosome, ind, axis=0)
            self.genom_length -= 1
        elif operation == "insert":
            ind = np.random.randint(0, len(self.chromosome), 1)[0]
            inserted_gene = np.array([np.random.randint(0, self.limits[i], 1)[0] for i in range(3)])
            self.chromosome = np.insert(self.chromosome, ind, inserted_gene, axis=0)
            self.genom_length += 1


class Phenotype():
    def __init__(self, genom: Genom, shape: np.ndarray, magnification=1):
        # magnificationは本質的ではないけれども、最後に画像を保存する際の画像サイズを指定する。
        shape = np.array(shape) * magnification
        self.morph = np.full(shape, 255, dtype=np.uint8) 
        self.genom = genom.chromosome * magnification# TODO: 直接渡しているのでよくない
        self.encode(shape, self.genom)

    def encode(self, shape, genom):
        mask = np.full(shape, True, dtype=np.bool)
        for gene in genom:
            mask = circle_mask(mask, *gene)
        self.morph *= mask        

    def overlap(self):
        total_circle_area = np.sum(self.genom[:, 2] ** 2 * np.pi)
        overlap = total_circle_area - np.sum(self.morph == 0)
        return overlap

    def circle_interfer(self):
        # overlapは大きな円に不利すぎた。面積ベースではなく、距離ベースで重なりを評価する。
        locations = self.genom[:, :2]
        rs = self.genom[:, 2]
        # 円同士の距離
        distances = np.sqrt(np.sum((np.expand_dims(locations, axis=0) - np.expand_dims(locations, axis=1)) ** 2, axis=2))
        # 円同士の半径
        arms = np.expand_dims(rs, axis=0) + np.expand_dims(rs, axis=1) - 2 * np.diag(rs) # 対各区成分を引いている
        interfer = arms - distances
        # 各円に対するinterferの値。標準化的なこと。円が大きいとペナルティが増大するのを避けたい。
        interfer = interfer / (rs + 0.1) # 半径0だとまずい
        # 十分に離れている場合は0
        interfer[interfer < - 0.01] = - 0.01
        return np.mean(interfer) # 円の数についても標準化することにした。

    def as_image(self):
        return self.morph
    
    def _mask(self, gene: np.ndarray):
        shape = self.morph.shape[:2]
        return circle_mask(shape, *gene)
    
    def show(self):
        plt.imshow(self.morph)
        plt.close()
    
    def save(self, path, title=""):
        plt.gray()
        plt.imshow(255 - self.morph, vmin = 0, vmax = 255)
        plt.title(title)
        plt.axis("off")
        plt.savefig(path)
        plt.close()
        
    def evaluate(self, target:np.ndarray):
        m = 100 
        n = 1000
        th = 0
        """
        評価値 = L2(downsample(blur(target)) - downsample(blur(generated))) + m * circle_overlap
        ダウンサンプルした後の目標画像との距離に加え、円同士が重なっている部分の面積をペナルティとして加える。（評価値は低いほど良い）
        overlapは、円の面積の単純な和 - エンコード画像の縁が描画されている範囲　で算出している。
        このため、円がキャンバスの外にはみ出ている場合もペナルティになってしまっている。
        それから、描画範囲は離散値なので、O(半径)くらいの誤差がでる。
        多少の重なりを許したいということもあるので（例えば500pix程度）threshold = 500 以上のoverlapのみをpneltyとした。
        L2(diff)は、最大255 * sqrt(ダウンサンプル後の画像サイズ~20*20) ~ 5000程度のオーダー
        overlapは、最大(半径~20) **2 * pi * (円の数~30) ~ 30000程度のオーダー
        mは円同士の重なりの重要性を評価する係数。
        L2(diff) の方を重要視してほしいので、m = 0.05程度に設定した。
        """
        ds_rate = min(self.morph.shape) // 32
        # 畳み込み -> 差分　と 差分 -> 畳み込み は等価なので、畳み込みを一回で済ました方がいい
        # down_sampled_morph = scipy.ndimage.gaussian_filter(self.morph, sigma=ds_rate)[::ds_rate, ::ds_rate]
        # down_sampled_target = scipy.ndimage.gaussian_filter(target, sigma=ds_rate)[::ds_rate, ::ds_rate]
        diff = scipy.ndimage.gaussian_filter(target - self.morph, sigma=ds_rate)[::ds_rate, ::ds_rate]
        overlap = self.circle_interfer()
        white_area_diff = np.sum(diff) / diff.size
        if random.randint(0,1000) == 0:
            print(f"L2: {np.linalg.norm(diff)}, overlap: {m * max(overlap, th)}, white_ratio: {np.abs(white_area_diff)}")
        return np.linalg.norm(diff) + m * max(overlap, th) +n * np.max(white_area_diff, 0)

class Family:
    def __init__(self, genom1, genom2):
        self.genom1 = genom1
        self.genom2 = genom2
        self.offspring1 = None
        self.offspring2 = None
        self.evaluation = [] # evalation of [genom1, genom2, offspring1, offspring2] 

    def _crossover(self):
        """
        交差によって子個体を生成する。
        """
        ch1 = self.genom1.chromosome
        ch2 = self.genom2.chromosome
        limits = self.genom1.getlimits()
        genom_length = min(len(ch1), len(ch2))
        cross_point_1 = np.random.randint(0, genom_length-1)
        cross_point_2 = np.random.randint(cross_point_1+1, genom_length)
        ofs1 = np.concatenate([ch1[:cross_point_1], ch2[cross_point_1:cross_point_2], ch1[cross_point_2:]])
        ofs2 = np.concatenate([ch2[:cross_point_1], ch1[cross_point_1:cross_point_2], ch2[cross_point_2:]])
        self.offspring1 = Genom(len(ofs1), limits=limits, chromosome=ofs1)
        self.offspring2 = Genom(len(ofs1), limits=limits, chromosome=ofs2)
            
    def breed(self, pm=0.05):
        """
        交差によって生成した子個体に突然変異を施し、結果を返す。
        """
        self._crossover()
        self.offspring1.mutate(pm)
        self.offspring2.mutate(pm)
        return [self.offspring1, self.offspring2]
    
    def _evaluate(self, target):
        if self.offspring1 is None:
            self._crossover()
        genom_list = [self.genom1, self.genom2, self.offspring1, self.offspring2]
        self.evaluation = np.array([Phenotype(g, shape=target.shape).evaluate(target) if g.evaluation is None else g.evaluation for g in genom_list]) # もしすでに計算していたら発現させるまでもない
    
    def _roulette(self, ind_list, select_num=1):
        if self.evaluation is []:
            self._evaluate()
        # 評価値は低いほうが良い、という方針を取っているが、ルーレット選択においては評価値と選択確率を比例させる必要がある。
        # そのため、最大評価値 + 最小評価値 - 自分の評価値 という変換によって順序を逆転させている。
        evaluation = self.evaluation[ind_list[0]] + self.evaluation[ind_list[-1]] - self.evaluation[ind_list]
        props = evaluation / np.sum(evaluation)
        return np.random.choice(ind_list, select_num, p=props)
    
    def mgg_change(self, target, pm=0.05):
        """
        世代間最小ギャップモデルに基づいて家族内での世代交代を行う。
        すなわち、交差・突然変異によって子を生成したのちに、評価値に基づくルーレット選択によって
        2個体を選び、返す。
        評価が入るので、targetを渡す必要がある: 構成を見直す必要がある
        """
        survivor = []
        # 交差を行う
        self.breed(pm=pm)
        genom_list = [self.genom1, self.genom2, self.offspring1, self.offspring2]
        self._evaluate(target)
        rank = np.argsort(self.evaluation)
        # エリート選択
        survivor.append(genom_list[rank[0]])
        luckey_ind = self._roulette(rank[1:], select_num=1)[0]
        survivor.append(genom_list[luckey_ind])
        # genomに評価値を格納しておく。再び評価されることがあれば計算を避けるため。
        survivor[0].set_evaluation(self.evaluation[rank[0]]) 
        survivor[1].set_evaluation(self.evaluation[luckey_ind])
        return survivor

    
class Generation:
    def __init__(self, generation_size=100, genom_length=10, genom_limits=[HEIGHT, WIDTH, WIDTH // 2],pm=0.05):
        self.size = generation_size
        self.genom_length = genom_length
        self.genom_limits = genom_limits
        self.genom_list = [Genom(genom_length, genom_limits) for i in range(generation_size)]
        self.evaluation = []
        self.elite_ind = []
        self.pm = pm
    
    def get_genom_list(self):
        return self.genom_list

    def get_evaluation(self):
        return self.evaluation

    def set_pm(self, pm):
        # 突然変異確率を決める
        self.pm = pm
    
    def evaluate(self, target):
        self.evaluation = np.array([Phenotype(g, shape=target.shape).evaluate(target) if g.evaluation is None else g.evaluation for g in self.genom_list])
        
    def mgg_change(self, target):
        """
        世代間最小ギャップモデルに基づく世代交代を行う。
        すなわち、ランダムに選ばれた2個体を交差し、子個体と合わせた4個体からルーレット選択により2個体を選択する。
        """
        parents_inds = random.sample(range(self.size), 2)
        family = Family(self.genom_list[parents_inds[0]], self.genom_list[parents_inds[1]])
        survivors = family.mgg_change(target, pm=self.pm)
        self.genom_list[parents_inds[0]] = survivors[0]
        self.genom_list[parents_inds[1]] = survivors[1]
        
    def select(self, elite_num):
        """
        self.genom_listの中からelite選択を行う。
        すなわち、evaluationの良い方からelite_num個だけ選択する。
        """
        elite_num = (elite_num // 2) * 2 # 偶数であることを担保したい
        self.elite_ind = np.argsort(self.evaluation)[:elite_num]
        self.genom_list = list(np.array(self.genom_list)[self.elite_ind]) # generation gap分の個体を淘汰
    

    def _breed_indivisual(self):
        """
        self.genom_listから二個体を非復元抽出し、交差を行う。
        交差の結果の子個体（2つ） を返す。
        """
        parents = random.sample(self.genom_list, 2)
        family = Family(*parents)
        return family.breed(self.pm)
    
    def breed(self, breed_num):
        """
        breed_numで指定した数だけ交差によって子個体をgenom_listに追加する。
        """
        while(len(self.genom_list) < self.size + breed_num):
            self.genom_list += self._breed_indivisual()

    def chief(self):
        """
        もっとも評価値の高いgenomとその評価値を返す。
        """
        chief_ind = np.argmin(self.evaluation) 
        return (self.genom_list[chief_ind], self.evaluation[chief_ind])

    # 以下、情報を表示するためのutility関数
    def summary(self):
        if self.evaluation == []:
            self.evaluate()
        mi, ma, ave = np.min(self.evaluation), np.max(self.evaluation), np.mean(self.evaluation)
        print(f"best: {mi:.1f}, worst: {ma:.1f}, ave: {ave:.1f}")
        genom_lengths = [len(g.chromosome) for g in self.genom_list]
        print(f"max circle num: {max(genom_lengths)}, min circle num: {min(genom_lengths)}")
        return mi, ma, ave
    
    def print_info(self):
        print("GENERATION INFO")
        print("---------------")
        print(f"generation size: {self.size}")
        print(f"circle num     : {self.genom_length}")
        print(f"canvas size    : {self.genom_limits[:2]}")
        print(f"Pm             : {self.pm}")
        
    def show_all(self, target, savepath="",):
        rownum = 5 # 適当
        colnum = (self.size + 5) // 5
        fig = plt.figure(figsize=[target.shape[1] * colnum //100, target.shape[0] * rownum // 100])

        ax0 = fig.add_subplot(rownum, colnum, 1)
        plt.axis('off')
        plt.gray()
        ax0.imshow(255-target)
        ax0.title.set_text("target")
        for i, (genom, evaluation) in enumerate(zip(self.genom_list, self.evaluation)):
            ax = fig.add_subplot(rownum, colnum, i+2)
            plt.axis('off')
            plt.gray()
            ax.imshow(255 - Phenotype(genom, shape=target.shape).as_image(), vmin = 0, vmax = 255)
            ax.title.set_text("{:.1f}".format(evaluation))
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath)
        else:
            fig.show()      
        plt.close()
