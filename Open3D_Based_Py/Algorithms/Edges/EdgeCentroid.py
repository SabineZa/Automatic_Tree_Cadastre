"""
Main task: Given 3D point clouds and several thresholds,
           generate edge candidates based the idea of 'centroid-gradient' in
          @article{xia2017fast,
                   title={A fast edge extraction method for mobile LiDAR point clouds},
                   author={Xia, Shaobo and Wang, Ruisheng},
                   journal={IEEE Geoscience and Remote Sensing Letters},
                   volume={14},
                   number={8},
                   pages={1288--1292},
                   year={2017},
                   publisher={IEEE}
                  }
NOTES：Based on the ablation study, the original method (in C++) is simplified and implemented in Python to improve its efficiency, generality, and robustness. 
In this version, the edge-index can be filtered directly, and its performance is close to (or even better than) the initial one.
@author: GeoMatrix Lab
"""
import numpy
import open3d as o3d
import numpy as np
import math
import os
import sys
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors

class Edge3DCentroid:
    def __init__(self):
        """
           Init parameters
        """
        self.pcd        = None  # input point clouds
        self.NPt        = 0     # input point cloud number
        self.Rnn        = 0.12  # r-neighbour sqrt(0.015), R-neighbour
        self.EI         = None  # Save point-wise edge-index
        self.GOrt       = None  # Grid-wise max-orientation
        self.Neighbours = None  # Neighbour system
        self.MaxNN      = 350   # Too many points, not good(edge index  over-smoothing)
        self.NoMaxT     = 0.95  # Decide the direction for No_MaxSuppression.
        self.GradientK  = 100   # Used in caculating gradients
        self.EICopy     = None
        np.seterr(divide='ignore', invalid='ignore')

    def SetPts(self, pc):
        self.pcd = pc
        self.NPt = len(self.pcd.points)
        self.EI  = np.zeros(self.NPt)


    # Caculate edge index by cpp-wrappers, much faster 
    # Time cost : from over 400s -> 69.07949757575989s  ->  15.45028567314148s after improving codes
    def CaculateEI_CPP(self):  
        self.Neighbours = cpp_neighbors.batch_query(np.asarray(self.pcd.points).astype(np.float32),
                                                    np.asarray(self.pcd.points).astype(np.float32),
                                                    [len(self.pcd.points)],
                                                    [len(self.pcd.points)],
                                                    radius = self.Rnn)
        self.Neighbours = self.Neighbours[:, :self.MaxNN]
        for i in range(self.NPt):
            idx = self.Neighbours[i, :]
            idx = idx[idx < self.NPt]
            CenterP = np.mean(np.asarray(self.pcd.points)[idx], axis = 0)  # Get center points
            l_dist = np.sqrt(
                    (np.asarray(self.pcd.points)[idx[len(idx) - 1]][0] - self.pcd.points[i][0]) ** 2 +
                    (np.asarray(self.pcd.points)[idx[len(idx) - 1]][1] - self.pcd.points[i][1]) ** 2 +
                    (np.asarray(self.pcd.points)[idx[len(idx) - 1]][2] - self.pcd.points[i][2]) ** 2)
            self.EI[i] = np.sqrt(np.sum((CenterP - self.pcd.points[i]) ** 2)) / l_dist

    # Get point-wise gradient
    def EdgeGradient_Simple(self):
        self.Neighbours = self.Neighbours[:, :self.GradientK]
        self.GOrt = np.zeros((self.NPt, 3))
        for i in range(self.NPt):
            idx = self.Neighbours[i, :]
            idx = idx[idx < self.NPt]
            gis = abs(self.EI[i] - self.EI[idx]) #gradient values
            maxDiffID = np.argmax(gis)
            # self.GI[i] = gis[maxDiffID] # if the gradient value is needed
            self.GOrt[i][0] = self.pcd.points[idx[maxDiffID]][0] - self.pcd.points[i][0]
            self.GOrt[i][1] = self.pcd.points[idx[maxDiffID]][1] - self.pcd.points[i][1]
            self.GOrt[i][2] = self.pcd.points[idx[maxDiffID]][2] - self.pcd.points[i][2]
            ds = np.sqrt(self.GOrt[i][0]**2 + self.GOrt[i][1]**2 + self.GOrt[i][2]**2)
            if ds < 0.00000001:
                ds = 0.00000001
            self.GOrt[i][0] = self.GOrt[i][0] / ds
            self.GOrt[i][1] = self.GOrt[i][1] / ds
            self.GOrt[i][2] = self.GOrt[i][2] / ds # Gradient direction

    #Non-Max Suppression based on gradients
    def No_MaxSuppression(self):
        self.EICopy = self.EI
        for i in range(self.NPt):
            idx = self.Neighbours[i, :]
            idx = idx[idx < self.NPt]
            dpt = np.asarray(self.pcd.points)[idx] - self.pcd.points[i]
            sum_of_rows = np.sqrt(np.sum(dpt**2, axis=1))
            NeigOrts    = dpt / sum_of_rows[:, np.newaxis]
            ######################################
            NeigOrts   = np.nan_to_num(NeigOrts)
            tileOrt    = np.tile(self.GOrt[i], (len(NeigOrts), 1))
            dotProduct = np.abs(tileOrt * NeigOrts).sum(axis=1)
            local_idx  = np.where(dotProduct > self.NoMaxT)
            local_idx  = idx[local_idx]
            if(self.EICopy[i] < self.EICopy[local_idx]).any():
                self.EI[i] = 0
