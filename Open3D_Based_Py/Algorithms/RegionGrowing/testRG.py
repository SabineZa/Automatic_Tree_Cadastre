import open3d as o3d
import Regions as RG
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    gPath = os.path.abspath(os.path.join(os.getcwd(), "../..")) ## Get Project path, Windows_10
    plyPath = os.path.join(gPath, 'TestData','fragment.ply')
    pcd = o3d.io.read_point_cloud(plyPath)
    # o3d.visualization.draw_geometries([pcd],
    #                                   zoom=0.7,
    #                                   front=[0.0, -0.5, -0.8499],
    #                                   lookat=[2.1813, 2.0619, 2.0999],
    #                                   up=[0.1204, -0.9852, 0.1215]) # Show original point clouds
    RGKNN = RG.RegionGrowing()
    RGKNN.SetDataThresholds(pcd,10.0) # the growing angle threshold is set to 10.0 degree
    RGKNN.RGKnn()  # Run region growing
    labels = RGKNN.ReLabeles()
    # Visualizer
    max_label = len(RGKNN.Clusters)
    print(f"point cloud has {max_label} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 1] = 1         # set to white for small clusters (label - 0 )
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.7,
                                      front=[0.0, -0.5, -0.8499],
                                      lookat=[2.1813, 2.0619, 2.0999],
                                      up=[0.1204, -0.9852, 0.1215])