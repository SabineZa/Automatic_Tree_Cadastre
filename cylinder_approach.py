# -*- coding: utf-8 -*-
"""
Automatic Tree Cadastre
This program automatically creates a tree cadastre from a point cloud.

Copyright (c) 2022-2023 Sabine Zagst (s.zagst@tum.de)

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math
import itertools

from Open3D_Based_Py.Algorithms.RegionGrowing import Regions as rg
import pyransac3d as pyrsc

# =============================================================================
# This function performs the cylinder approach:
# 1. Get point cloud slice using PassThrough filter
# 2. Compute Normals
# 3. Get a cluster for every tree trunk using region growing
# 4. Fit a cylinder in every tree trunk cluster with RANSAC to get parameters e.g. DBS
# Input: nonGround_o3d -> PointCloud object of open3D; visualize -> boolean
# Output: dbh
# =============================================================================


def cylinder_approach(nonGround_o3d, visualize=False):

    # TODO: nicht-Bäume-Cluster entfernen vor dem RANSAC (Eigenwerte, Outlier Statistik, ...)

    # %% *************** SEGMENTATION - PASSTHROUGH ****************
    # Get a point cloud slice
    # Source: https://betterprogramming.pub/point-cloud-filtering-in-python-e8a06bbbcee5#830e

    # Create bounding box:
    bounds = [
        [-math.inf, math.inf],
        [-math.inf, math.inf],
        [1.8, 2.5],
    ]  # set the bounds
    bounding_box_points = list(itertools.product(*bounds))  # create limit points
    bounding_box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(bounding_box_points)
    )  # create bounding box object

    # Crop the point cloud using the bounding box:
    nonGround_cropped_o3d = nonGround_o3d.crop(bounding_box)
    # nonGround_cropped_array = np.asarray(nonGround_cropped_o3d.points)

    # output for user
    print("PassThrough filter successful.")

    # Display the cropped point cloud:
    if visualize:
        o3d.visualization.draw_geometries([nonGround_cropped_o3d])

    # %% ****************** FEATURES - NORMALS *************************
    # http://www.open3d.org/docs/latest/python_api/open3d.geometry.PointCloud.html?highlight=normals#open3d.geometry.PointCloud.estimate_normals
    # http://www.open3d.org/docs/latest/python_api/open3d.geometry.KDTreeSearchParamRadius.html

    nonGround_cropped_o3d.estimate_normals(
        o3d.geometry.KDTreeSearchParamRadius(0.2)
    )  # [m]
    # Orientierung an Baumstammdurchmesser, nicht zu große Aufloesung noetig, da nur der Stamm interessiert
    nonGround_cropped_array = np.asarray(nonGround_cropped_o3d.points)
    # nonGround_cropped_normals = np.asarray(nonGround_cropped_o3d.normals)

    # output for user
    print("Normal estimation successful.")

    # %% ********************** SEGMENTATION - REGION GROWING *****************
    # Source: https://github.com/GeoVectorMatrix/Open3D_Based_Py

    RGKNN = rg.RegionGrowing()

    # Parameter settings
    RGKNN.SetDataThresholds(
        nonGround_cropped_o3d, 180.0
    )  # input point cloud and growing angle threshold. 180 degrees, since they are round tree trunks.
    RGKNN.rKnn = 50  # region growing using k-neighbour
    RGKNN.rRnn = 0.5  # region growing using r-neighbour
    RGKNN.minCluster = 500  # minimal cluster size
    # If the input point cloud does not have normals, they are automatically
    # calculated before region growing using
    # self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.nRnn, max_nn=self.nKnn))
    # For this purpose, the following parameters can be set:
    if len(nonGround_cropped_o3d.normals) != len(nonGround_cropped_o3d.points):
        RGKNN.nKnn = 20  # normal estimation using k-neighbour
        RGKNN.nRnn = 0.2  # normal estimation using r-neighbour

    RGKNN.RGKnn()  # Run region growing
    labels = RGKNN.ReLabeles()
    cluster_indices = RGKNN.Clusters

    # Visualizer
    if visualize:
        max_label = len(cluster_indices)
        print(f"Region growing resulted in {max_label} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 1] = 1  # set to white for small clusters (label - 0 )
        nonGround_cropped_o3d.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([nonGround_cropped_o3d])

    # output for user
    print("Region growing successful.")

    # %% ***************** SEGMENTATION - RANSAC ************************
    # RANSAC with cylinder model for tree trunk detection
    # For this pyRANSAC-3D is used: https://pypi.org/project/pyransac3d/
    # https://leomariga.github.io/pyRANSAC-3D/api-documentation/cylinder/

    cylinder = pyrsc.Cylinder()
    cylinder_inliers = []
    trunk_radius = []
    cluster_o3d = o3d.geometry.PointCloud()
    # fit a cylinder in every cluster
    for i in range(len(cluster_indices)):
        cluster = np.take(nonGround_cropped_array, cluster_indices[i], axis=0)
        center, axis, radius, inlier_indices = cylinder.fit(
            cluster, thresh=0.1, maxIteration=5000
        )

        cylinder_inliers.append(np.take(cluster, inlier_indices, axis=0))
        trunk_radius.append(radius)

        # Visualization of current cylinder inliers
        if visualize:
            cluster_o3d.points = o3d.utility.Vector3dVector(cylinder_inliers[i])
            o3d.visualization.draw_geometries([cluster_o3d])

    return trunk_radius * 2
