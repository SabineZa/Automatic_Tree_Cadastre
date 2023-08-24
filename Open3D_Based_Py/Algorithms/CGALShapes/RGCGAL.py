# -*- coding: utf-8 -*-
"""
Created on Wed May 12 16:26:20 2021

@author: GeoFunny
"""
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
from CGAL.CGAL_Shape_detection import region_growing as RG_CGAL
import numpy as np
import open3d as o3d

class RGPlanes(object):
    """
    Given point clouds, detect as many planes as possible; Index of points will be consistent.
    """
    def __init__(self, inPts):
        # dict: from id to pts.
        self.pts             = inPts  # numpy array
        self.pt_labels       = np.zeros(len(self.pts))
        self.plane_map       = None
        self.nb_planes       = 0
        self.Pt_Labels       = -1*np.zeros(len(inPts)).astype(int)
        #
        self.pcd             = o3d.geometry.PointCloud()
        self.pcd.points      = o3d.utility.Vector3dVector(inPts)
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))       
   
    def RG_all(self): 
        CGALPts = Point_set_3()
        CGALPts.add_normal_map()
        for i in range(len( self.pts)):
            CGALPts.insert(Point_3(self.pts[i][0], self.pts[i][1], self.pts[i][2]),
                           Vector_3(self.pcd.normals[i][0],self.pcd.normals[i][1],self.pcd.normals[i][2]))  
        self.plane_map = CGALPts.add_int_map("plane_index")
        self.nb_planes = RG_CGAL(CGALPts, self.plane_map, min_points=100)
        #
        for i in range(len(self.pts)):
            self.Pt_Labels[i] = CGALPts.int_map('plane_index').get(i)
        
