import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import EdgeCentroid as ECG
import time

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(".\\TestData\\fragment.ply") # It takes about 45s
    EdgePts = ECG.Edge3DCentroid()
    EdgePts.SetPts(pcd)
    #
    time_start = time.time()
    EdgePts.CaculateEI_CPP()      # Edge index
    EdgePts.EdgeGradient_Simple() # Gradients
    EdgePts.No_MaxSuppression()   # No_MaxSuppression
    time_end = time.time()
    #
    print('time cost',time_end-time_start,'s')
    print(np.max(EdgePts.EI))
    print(np.min(EdgePts.EI))
    
    # Hard thresholding 
    edgeIdx = np.where(EdgePts.EI > 0.22)
    le  = [list(i) for i in edgeIdx]
    le  = le[0]
    pcdEdge = pcd.select_by_index(le)
    colors = np.zeros((len(le), 3))
    pcdEdge.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcdEdge],
                                      zoom=0.7,
                                      front=[0.0, -0.5, -0.8499],
                                      lookat=[2.1813, 2.0619, 2.0999],
                                      up=[0.1204, -0.9852, 0.1215])