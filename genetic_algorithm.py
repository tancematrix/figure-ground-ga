# coding:utf-8
from matplotlib import pyplot as plt
import numpy as np
from typing import NamedTuple
import scipy.ndimage
import random
import pickle

HEIGHT, WIDTH = 150, 90

# 補助的な関数
def circle_mask(shape, cx, cy, r):
    x, y = np.ogrid[:shape[0], :shape[1]]
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    mask = r2 > r ** 2
    return mask

class Genom:
    def __init__(self, genom_length, random=True, limits=[HEIGHT, WIDTH, WIDTH // 2], chromosome=None):
        self.genom_length = genom_length
        self.gene_length = 3 # ハードコーディング
        self.chromosome = np.zeros([self.genom_length, self.gene_length])
        self.limits = limits
        if chromosome is not None:
            self.chromosome = chromosome
        elif random:
            self.randomize()
    
    def randomize(self):
        # ハードコーディング
        self.chromosome[:,0] = np.random.randint(0, self.limits[0], self.genom_length)
        self.chromosome[:,1] = np.random.randint(0, self.limits[1], self.genom_length)
        self.chromosome[:,2] = np.random.randint(0, self.limits[2], self.genom_length) # 円の半径
    
    def mutate(self, pm):
        whichwillbechanged = np.random.choice([True, False], self.chromosome.shape, p=[pm, 1-pm])
        self.chromosome[whichwillbechanged[:,0], 0] = np.random.randint(0, HEIGHT, np.sum(whichwillbechanged[:,0]))
        self.chromosome[whichwillbechanged[:,1], 1] = np.random.randint(0, WIDTH, np.sum(whichwillbechanged[:,1]))
        self.chromosome[whichwillbechanged[:,2], 2] = np.random.randint(0, WIDTH//2, np.sum(whichwillbechanged[:,2]))

class Phenotype():
    def __init__(self, genom: Genom, shape):
        self.morph = np.full(shape, 255)
        self.genom = genom.chromosome # TODO: 直接渡しているのでよくない
        self.encode()

    def encode(self):
        mask = np.multiply.reduce(np.apply_along_axis(self._mask, 1, self.genom))
        self.morph = self.morph * mask

    def as_image(self):
        return self.morph
    
    def _mask(self, gene: np.ndarray):
        shape = self.morph.shape[:2]
        return circle_mask(shape, *gene)
    
    def show(self):
        plt.imshow(self.morph)
    
    def save(self, path, title=""):
        plt.gray()
        plt.imshow(self.morph)
        plt.title(title)
        plt.savefig(path)
        
    def evaluate(self, target:np.ndarray):
        ds_rate = min(self.morph.shape) // 10
        down_sampled_morph = scipy.ndimage.gaussian_filter(self.morph, sigma=ds_rate)[::ds_rate, ::ds_rate]
        down_sampled_target = scipy.ndimage.gaussian_filter(target, sigma=ds_rate)[::ds_rate, ::ds_rate]
        return np.linalg.norm(down_sampled_morph - down_sampled_target)

class Family:
    def __init__(self, genom1, genom2):
        self.genom1 = genom1
        self.genom2 = genom2
        self.offspring1 = None
        self.offspring2 = None
        
    def _crossover(self):
        ch1 = self.genom1.chromosome
        ch2 = self.genom2.chromosome
        genom_length = len(ch1)
        cross_point_1 = np.random.randint(0, genom_length-1)
        cross_point_2 = np.random.randint(cross_point_1+1, genom_length)
        ofs1 = np.concatenate([ch1[:cross_point_1], ch2[cross_point_1:cross_point_2], ch1[cross_point_2:]])
        ofs2 = np.concatenate([ch2[:cross_point_1], ch1[cross_point_1:cross_point_2], ch2[cross_point_2:]])
        self.offspring1 = Genom(genom_length, chromosome=ofs1)
        self.offspring2 = Genom(genom_length, chromosome=ofs2)
            
    def breed(self, pm=0.05):
        self._crossover()
        self.offspring1.mutate(pm)
        self.offspring2.mutate(pm)
        return [self.offspring1, self.offspring2]

    
class Generation:
    def __init__(self, generation_size=100, genom_length=10, genom_limits=[HEIGHT, WIDTH, WIDTH // 2],pm=0.05):
        self.size = generation_size
        self.genom_length = genom_length
        self.genom_limits = genom_limits
        self.genom_list = [Genom(genom_length, genom_limits) for i in range(generation_size)]
        self.evaluation = []
        self.elite_ind = []
        self.pm = pm
    
    def set_pm(self, pm):
        # 突然変異確率を決める
        self.pm = pm
    
    def evaluate(self, target):
        self.evaluation = np.array([Phenotype(g, shape=target.shape).evaluate(target) for g in self.genom_list])
        
    def select(self, elite_num):
        elite_num = (elite_num // 2) * 2 # 偶数であることを担保したい
        self.elite_ind = np.argsort(self.evaluation)[:elite_num]
        self.genom_list = list(np.array(self.genom_list)[self.elite_ind]) # generation gap分の個体を淘汰
    

    def _breed_indivisual(self):
        parents = random.sample(self.genom_list, 2)
        family = Family(*parents)
        return family.breed(self.pm)
    
    def breed(self):
        while(len(self.genom_list) < self.size):
            self.genom_list += self._breed_indivisual()

    def chief(self):
        chief_ind = np.argmin(self.evaluation) 
        return (self.genom_list[chief_ind], self.evaluation[chief_ind])

    def summary(self):
        mi, ma, ave = np.min(self.evaluation), np.max(self.evaluation), np.mean(self.evaluation)
        print(f"best: {mi:.1f}, worst: {ma:.1f}, ave: {ave:.1f}")
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
        ax0.imshow(target)
        ax0.title.set_text("target")
        for i, (genom, evaluation) in enumerate(zip(self.genom_list, self.evaluation)):
            ax = fig.add_subplot(rownum, colnum, i+2)
            plt.axis('off')
            plt.gray()
            ax.imshow(Phenotype(genom, shape=target.shape).as_image())
            ax.title.set_text("{:.1f}".format(evaluation))
        if savepath:
            plt.savefig(savepath)
        else:
            fig.show()      

