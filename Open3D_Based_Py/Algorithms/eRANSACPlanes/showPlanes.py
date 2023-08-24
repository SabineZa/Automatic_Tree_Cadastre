# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:01:35 2021

@author: GeoFunny
"""


import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from ERANSACPlanes import *

if __name__ == '__main__':
    Pts = threeDataset.pts['cloud_bin_1']
    mPlanes = ERansacPlanes(Pts)
    mPlanes.eransac_all()
    # Visualizer
    planeLabels = mPlanes.Pt_Index_Planes + 1
    max_label   = max(mPlanes.Pt_Index_Planes) + 1
    pcolors = plt.get_cmap("tab20")(planeLabels / (max_label if max_label > 0 else 1))
    pcolors[labels < 0] = 1   # set to white for other points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(Pts)
    pcd.colors = o3d.utility.Vector3dVector(pcolors[:, :3])
    o3d.visualization.draw_geometries([pcd])