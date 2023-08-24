"""
Please cite the following paper (related to this topic), if this code helps you in your research.
   @article{xia2020geometric,
            title={Geometric primitives in LiDAR point clouds: A review},
            author={Xia, Shaobo and Chen, Dong and Wang, Ruisheng and Li, Jonathan and Zhang, Xinchang},
            journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
            volume={13},
            pages={685--707},
            year={2020},
            publisher={IEEE}
           }
    Parts of codes are from https://github.com/CGAL/cgal-swig-bindings/blob/main/examples/python/Shape_detection_example.py
"""                 
from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Vector_3
from CGAL.CGAL_Point_set_3 import Point_set_3
from CGAL.CGAL_Shape_detection import *
import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt

# In this demo, CGAL and Open3D are combined
# CGAL can provide backbone methods and Open3D can be used for visualization

if __name__ == "__main__":
    gPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) ## Get Project path, Windows_10
    datafile = os.path.join(gPath, 'TestData','fragment.ply')
    points = Point_set_3(datafile)
    # Prepare data for open3d
    npPt = np.zeros((points.size(), 3))
    for i in range(points.size()):
        ar = np.array([points.point(i).x(),points.point(i).y(),points.point(i).z()])
        npPt[i] = ar.copy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(npPt)
    #o3d.visualization.draw_geometries([pcd])  # show original point clouds.
    # CGAL plane detection by the region growing
    print(points.size(), "points read")
    print("Detecting planes with region growing (sphere query)")
    plane_map = points.add_int_map("plane_index")
    nb_planes = region_growing(points, plane_map, min_points=100)
    print(nb_planes, "planes(s) detected")
    #
    labels  = np.zeros(points.size()).astype(int)
    PlaneList = [[] for i in range(nb_planes+1)]  # index -1 for others
    for i in range(points.size()):
        labels[i] = points.int_map('plane_index').get(i)
        PlaneList[labels[i]+1].append(i)

    print(len(PlaneList))
    # Visualizer
    max_label = len(PlaneList)
    print(f"point cloud has {max_label} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 1  # set to white for small clusters (label - 0 )
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.7,
                                      front=[0.0, -0.5, -0.8499],
                                      lookat=[2.1813, 2.0619, 2.0999],
                                      up=[0.1204, -0.9852, 0.1215])
