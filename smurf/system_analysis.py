from matplotlib import pyplot as plt
import json
import pandas as pd
from shapely.geometry.point import Point
from shapely import affinity
from matplotlib.patches import Polygon
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

def shapely_ellipse(center,dims,scale,angle):
    circ = Point(center).buffer(1)
    elld = affinity.scale(circ, dims[0], dims[1])
    ellr = affinity.rotate(elld, angle)
    ells = affinity.scale(ellr, scale[0], scale[1])
    return ells

def create_elipse(x,y,ax,color,n_std):
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    width = (np.sqrt(1 + pearson))
    height = (np.sqrt(1 - pearson))
    x_center = np.mean(x)
    y_center = np.mean(y)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    ellipse_obj = shapely_ellipse((x_center,y_center),(width,height),(scale_x, scale_y),45)
    verts1 = np.array(ellipse_obj.exterior.coords.xy)
    patch = Polygon(verts1.T, facecolor=color,edgecolor='black', alpha=0.15)
    return ellipse_obj, ax.add_patch(patch)

def estimate_center(x_all,y_all):
        X = np.concatenate((x_all.reshape(-1, 1), y_all.reshape(-1, 1)), axis=1)
        km = KMeans(n_clusters=1).fit(X)
        closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)
        return list(x_all).pop(closest[0]), list(y_all).pop(closest[0]), closest[0]

def get_inter(e1,e2):
    return e1.intersection(e2).area/e2.area

##########################################################################################
class smurf_system_analysis():
    def __init__(self,n_systems=4,ref_system=0,in_file='results/smurf_scores.json'):
        self.n_systems = n_systems
        self.ref_system = ref_system
        self.in_file = in_file

    def load_standardized_scores(self,estimates_file='smurf/standardize_estimates.txt'):
        stand_in = open(self.in_file, "r")
        metric_scores = json.load(stand_in)
        estimates = pd.read_csv(estimates_file, header=None)
        sem_ind = list(estimates[0]).index('SPARCS')
        qual_ind = list(estimates[0]).index('SPURTS')
        gram_ind = list(estimates[0]).index('MIMA')
        self.stand_SPARCS = (metric_scores["SPARCS"] - estimates.loc[sem_ind, 1]) / estimates.loc[sem_ind, 2]
        self.stand_SPURTS = (metric_scores["SPURTS"] - estimates.loc[qual_ind, 1]) / estimates.loc[qual_ind, 2]
        self.stand_MIMA = (metric_scores["MIMA"] - estimates.loc[gram_ind, 1]) / estimates.loc[gram_ind, 2]

    def compute_grammar_penalities(self,outlier_thres=-1.96):
        penalties = []
        for i in range(0,self.n_systems):
            gram_penalty = self.stand_MIMA[i::self.n_systems] - outlier_thres
            gram_penalty[gram_penalty > 0] = 0
            penalties.append(np.sum(gram_penalty))
        return penalties

    def print_ellipse_intersections(self):
        iter = list(np.arange(self.n_systems))
        cand_systems = iter[:self.ref_system] + iter[self.ref_system + 1:]
        intersections = []
        for i in cand_systems:
            print('Intersection ' + str(i) + ' = ' + str(get_inter(self.ellipse[self.ref_system],self.ellipse[i])))
        return intersections

    def generate_plot(self,colors,out_file='results/system_plot.png',num_random_pts=100,seed=10,n_std=1.15):
        assert len(colors) == self.n_systems
        random.seed(seed)
        fig = plt.figure(0)
        ax = fig.add_subplot(111, aspect='equal')
        self.ellipse = []
        center_x = []
        center_y = []
        estimate_set = []

        for i in range(0,self.n_systems):
            x_all = self.stand_SPARCS[i::self.n_systems]
            y_all = self.stand_SPURTS[i::self.n_systems]
            estimate_x, estimate_y, estimate_index = estimate_center(x_all,y_all)
            center_x.append(estimate_x)
            center_y.append(estimate_y)
            estimate_set.append(estimate_index)

        rand_set = []
        for _ in range(0,num_random_pts):
            num_pts = int(len(self.stand_SPARCS)/self.n_systems)
            rand_set.append(random.choice([i for i in range(0,num_pts) if i not in estimate_set+rand_set]))

        for i in range(0, self.n_systems):
            x_all = self.stand_SPARCS[i::self.n_systems]
            y_all = self.stand_SPURTS[i::self.n_systems]
            x = [x_all[j] for j in rand_set]
            y = [y_all[j] for j in rand_set]
            self.ellipse.append(create_elipse(x_all,y_all, ax, colors[i], n_std)[0])
            ax.scatter(x,y,5,c=colors[i])

        for i in range(0,self.n_systems):
            ax.scatter(center_x[i],center_y[i],s=60,c=colors[i],marker='^',edgecolors='black')

        self.print_ellipse_intersections()

        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        plt.xlabel('SPARCS (Semantic Score)')
        plt.ylabel('SPURTS (Style Score)')
        plt.savefig(out_file)
        plt.show()