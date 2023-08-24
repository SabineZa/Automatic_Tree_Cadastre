# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:12:41 2021

@author: GeoFunny
"""
import numpy as np
import open3d as o3d

class ERansacPlanes(object):
    """
    Given point clouds, detect as many planes as possible; Index of points will be consistent.
    """
    def __init__(self, inPts):
        # dict: from id to pts.
        self.pts             = inPts  # numpy array
        self.pcd             = o3d.geometry.PointCloud()
        self.Pt_Index_Planes = -1*np.ones(len(self.pts))
        self.PlaneList       = []  #Detected plane parameters
        self.d_t             = 0.01
        self.s_n             = 3
        self.max_iter        = 10000
        self.min_n_plane     = max(50, int(0.01*len(self.pts)))  # minimal number of points per plane
        self.max_n_planes    = 2
   
    def eransac_all(self): 
        while 1:
          current_index = np.where(self.Pt_Index_Planes < 0)
          self.pcd.points      = o3d.utility.Vector3dVector(self.pts[current_index])
          plane_model, inliers = self.pcd.segment_plane(distance_threshold= self.d_t,
                                                        ransac_n= self.s_n ,
                                                        num_iterations= self.max_iter)
          if len(inliers) > self.min_n_plane:
              self.PlaneList.append(plane_model)  # Add to plane list
              real_index = current_index[0][inliers] # Real index in the input point clouds
              self.Pt_Index_Planes[real_index] = len(self.PlaneList) # Plane index    
          else:
              break
          if self.max_n_planes < len( self.PlaneList):
              break
