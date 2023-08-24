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

from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import time

import CSF

# https://pypi.org/project/cloth-simulation-filter/


def process_ground(
    cloud_array: np.ndarray,
    mode: str,
    step: float = 1.0,
    visualize: bool = False,
    save: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """To prepare the point cloud for the next processing steps, the ground points are first
     separated from the rest of the point cloud. After that, the height values of the non-ground
     points are adjusted depending on the mode.

    Parameters
    ----------
    cloud_array : np.ndarray
        3D point cloud (x,y,z)
    mode : str
        Mode to adjust the height values of the non-ground points. 3 available modes:
        mean: Set ground as zero level by using the mean-z-value of all ground points.
        median: Set ground as zero level by using the median-z-value of all ground points.
        ground_grid: Get all non-ground points to the same height level by subtracting the median
        value corresponding to the ground grid cell.
    step : float, optional
        Size of the ground cell in ground_grid mode, by default 1.0 [m]
    visualize : bool, optional
        If true, resulting point clouds are visualized, by default False
    save : bool, optional
        If true, saves a plot of the ground grid to Images/, by default False

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Array with ground points (x,y,z), array with non-ground points (x,y,z_new) and raster grid
        containing the median of the z-values of all points in each raster cell. [x,y]
    """

    # Separate ground from point cloud using the Cloth Simulation Filter
    start = time.time()
    ground_array, nonGround_array = separate_ground(cloud_array, visualize)
    end = time.time()
    print("CSF took {:5.3f}s to complete.\n".format(end - start), end=" ")

    if mode == "mean":
        # Set ground as zero level by using the mean-z-value of all ground points
        mean = np.mean(ground_array, axis=0, dtype=float)
        nonGround_array[:, 2] = nonGround_array[:, 2] - mean[2]

    if mode == "median":
        # Set ground as zero level by using the median-z-value of all ground points
        median = np.mean(ground_array)
        nonGround_array[:, 2] = nonGround_array[:, 2] - median[2]

    if mode == "ground_grid":
        start = time.time()
        grid = median_ground_grid(ground_array, nonGround_array, step, save)
        nonGround_array = uniform_height_level(nonGround_array, grid, step)
        end = time.time()
        print("It took {:5.3f}s to get an uniform height level. \n".format(end - start), end=" ")

    else:
        print("Select a valid ground processing mode.")

    # Save and visualize the point cloud with non-ground points after updating its height-values
    cloud_updatet = o3d.geometry.PointCloud()
    cloud_updatet.points = o3d.utility.Vector3dVector(nonGround_array)

    if visualize:
        o3d.visualization.draw_geometries([cloud_updatet])

    return ground_array, nonGround_array, grid


def separate_ground(
    cloud_array: np.ndarray, visualize: bool = False
) -> tuple[np.ndarray, np.ndarray]:

    """Separate ground from point cloud using the Cloth Simulation Filter.
    Github Repository:  https://github.com/jianboqi/CSF

    Parameters
    ----------
    cloud_array : np.ndarray
        Point cloud (x,y,z)
    visualize : bool, optional
        Should the resulting ground be visualized?, by default False

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Ground array containing the ground points (x,y,z) and
        nonGround array containing the remaining points (x,y,z)
    """

    csf = CSF.CSF()
    csf.setPointCloud(cloud_array)

    # Parameter settings
    # Source Parameter-Definitions:
    # http://ramm.bnu.edu.cn/researchers/wumingzhang/english/default_contributions.htm
    csf.params.bSloopSmooth = False  # whether the post-processing is needed or not. This is used to handle the steep slopes. (If steep slopes, such as river bank, exists, this value should be set as 1; If the terrain is flat, then this value can be set as 0).
    csf.params.time_step = 0.65  # time step for each iteration. Smaller the value is, more accurate the result may be, but more computing-time is needed. 0.65 is applicable to most of situations.
    csf.params.class_threshold = 0.5  # a threshold used to classify the original point cloud into ground and non-ground parts based on the distances between original point cloud and the simulated particles of cloth. 0.5 is also adapted to most of situations.
    csf.params.cloth_resolution = 0.5  # the horizon space between two particles (i.e., the resolution of cloth grid). This parameters should be smaller than the point spacing of original lidar point cloud. Usually, this can be set as 1/3 or point spacing.
    csf.params.rigidness = 3  # greater the value is, harder the cloth will be (usually this value can be 1,2 or 3)
    csf.params.interations = 500  # the maximum iteration times of cloth simulation. 500 is enough for most of situations.

    # do filtering and get ground and non-ground indexes
    groundIndexes = CSF.VecInt()
    nonGroundIndexes = CSF.VecInt()
    # Interesting question: Where is the Cloth exported to?
    csf.do_filtering(groundIndexes, nonGroundIndexes, exportCloth=False)
    groundIndexes = np.asarray(groundIndexes)
    # necessary to get Array of int32 instead of VecInt object of CSF module
    nonGroundIndexes = np.asarray(nonGroundIndexes)

    # create clouds as numpy array and open3d point cloud for ground and non-ground points
    nonGround_array = np.take(cloud_array, nonGroundIndexes, axis=0)
    ground_array = np.take(cloud_array, groundIndexes, axis=0)

    nonGround_o3d = o3d.geometry.PointCloud()
    ground_o3d = o3d.geometry.PointCloud()
    nonGround_o3d.points = o3d.utility.Vector3dVector(nonGround_array)
    ground_o3d.points = o3d.utility.Vector3dVector(ground_array)

    # output for user
    print("CSF done.")
    print(f"Number of non-ground points: {nonGround_array.shape[0]}")
    print(f"Number of ground points: {ground_array.shape[0]}")

    # save pointclouds
    # o3d.io.write_point_cloud("ground.pcd", ground_o3d, write_ascii=False)
    # o3d.io.write_point_cloud("non-ground.pcd", nonGround_o3d, write_ascii=False)

    if visualize:
        o3d.visualization.draw_geometries([ground_o3d])
        o3d.visualization.draw_geometries([nonGround_o3d])

    return ground_array, nonGround_array


def median_ground_grid(
    ground_array: np.ndarray, nonGround_array: np.ndarray, step: float, save: bool
) -> np.ndarray:
    """This function rasterizes the ground point cloud in the x-y plane. Each raster cell has
    the size step x step [m]. Then, for each raster cell, the median of the z-values of
    all points contained in it is calculated.

    Parameters
    ----------
    ground_array : np.ndarray
        Ground point cloud (x,y,z)
    nonGround_array : np.ndarray
        Point cloud with non-ground points
    step : float
        Size of a raster cell [m]
    save : bool
        If true, saves a plot of the ground grid to Images/.

    Returns
    -------
    np.ndarray
        Raster grid containing the median of the z-values of all points in each raster cell. [x,y]
    """

    # Limits of the non-ground point cloud, since what matters in the end is its extent.
    # The points of the non-ground point cloud should be brought to a uniform height level.
    xLimits = np.round(
        [np.min(nonGround_array[:, 0]), np.max(nonGround_array[:, 0])], 3
    )
    yLimits = np.round(
        [np.min(nonGround_array[:, 1]), np.max(nonGround_array[:, 1])], 3
    )

    # Initialize variable last-median with median of all ground points.
    # Contains later the last calculated median value.
    last_median = np.median(ground_array[:, 2])

    # Ground raster in x-y-layer
    xRaster = np.round(
        np.arange(xLimits[0], xLimits[1] + step, step), 3
    )  # arange excludes stop-value -> + step
    yRaster = np.round(np.arange(yLimits[0], yLimits[1] + step, step), 3)
    grid = np.full((len(xRaster), len(yRaster)), last_median)

    # Fill the grid with the median z-value of the ground points in each grid cell
    for i in range(len(xRaster) - 1):
        # points with the cell's x-values
        xColumn_points = np.round(
            ground_array[np.round(ground_array[:, 0], 3) >= xRaster[i], :], 3
        )
        xColumn_points = xColumn_points[xColumn_points[:, 0] < xRaster[i + 1], :]

        for j in range(len(yRaster) - 1):
            # points with the cell's x and y-values
            xColumn_points = xColumn_points[xColumn_points[:, 1] >= yRaster[j], :]
            cell_points = xColumn_points[xColumn_points[:, 1] < yRaster[j + 1], :]
            if np.size(cell_points) > 0:
                z_median = np.median(cell_points[:, 2])
                last_median = z_median
            else:
                z_median = last_median
            grid[i, j] = z_median

    c = plt.imshow(
        np.rot90(grid, 1, (0, 1)),
        cmap="summer",
        extent=[xLimits[0], xLimits[1], yLimits[0], yLimits[1]],
    )
    plt.colorbar(c, label="Höhe [m]")
    plt.title("Bodenraster mit medialen Höhenwerten")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    if save:
        p = Path("Images/")
        p.mkdir(parents=True, exist_ok=True)
        plt.savefig(p / "bodenraster_plot.jpg")
        print("Plot bodenraster_plot.jpg is saved to Images/.")
    plt.show()
    plt.close()

    return grid


def uniform_height_level(
    nonGround_array: np.ndarray, grid: np.ndarray, step: float
) -> np.ndarray:
    """Get all non-ground points to the same height level by subtracting
    the median value corresponding to the grid cell.

    Parameters
    ----------
    nonGround_array : np.ndarray
        Point cloud with non-ground points (x,y,z).
    grid : np.ndarray
        Raster grid [x,y] containing the median of the z-values of all points in each raster cell.
    step : float
        Size of a raster cell. [m]

    Returns
    -------
    np.ndarray
        Point cloud with non-ground points (x,y,z_new) reduced to the same hight level.
    """

    nonGround_updatet = np.zeros((1, 3))
    # Limits of the non-ground point cloud, since what matters in the end is its extent.
    # The points of the non-ground point cloud should be brought to a uniform height level.
    xLimits = np.round(
        [np.min(nonGround_array[:, 0]), np.max(nonGround_array[:, 0])], 3
    )
    yLimits = np.round(
        [np.min(nonGround_array[:, 1]), np.max(nonGround_array[:, 1])], 3
    )

    # Ground raster in x-y-layer
    xRaster = np.round(
        np.arange(xLimits[0], xLimits[1] + step, step), 3
    )  # arange excludes stop-value -> + step
    yRaster = np.round(np.arange(yLimits[0], yLimits[1] + step, step), 3)

    # Get all non-ground points to the same height level by subtracting the
    # median value corresponding to the grid cell.
    for i in range(len(xRaster) - 1):
        # points with the cell's x-values
        xColumn_points = np.round(
            nonGround_array[np.round(nonGround_array[:, 0], 3) >= xRaster[i], :], 3
        )
        xColumn_points = xColumn_points[xColumn_points[:, 0] < xRaster[i + 1], :]

        for j in range(len(yRaster) - 1):
            # points with the cell's x and y-values
            xColumn_points = xColumn_points[xColumn_points[:, 1] >= yRaster[j], :]
            cell_points = xColumn_points[xColumn_points[:, 1] < yRaster[j + 1], :]
            if np.size(cell_points) > 0:
                cell_points[:, 2] = cell_points[:, 2] - grid[i, j]
                nonGround_updatet = np.concatenate(
                    (nonGround_updatet, cell_points), axis=0
                )
            else:
                continue
    nonGround_updatet = np.delete(nonGround_updatet, 0, 0)
    return nonGround_updatet


def transform_back_z(
    x_y_coords: np.ndarray,
    ground_grid: np.ndarray,
    nonGround_array: np.ndarray,
    step: float,
) -> np.ndarray:
    """This function determines the matching z-coordinate to the given x- and y-coordinates using the ground grid.

    Parameters
    ----------
    x_y_coords : np.ndarray
        Array with x- and y-coordinates (x, y) [m]
    ground_grid : np.ndarray
        Raster grid containing the median of the z-values of all points in each raster cell. [x,y]
    nonGround_array : np.ndarray
        Point cloud with non-ground points (x,y,z).
    step : float
        Size of a raster cell [m]

    Returns
    -------
    np.ndarray
        3D coordinates (x, y, z) [m]
    """

    # Limits of the non-ground point cloud, since what matters in the end is its extent.
    xLimits = np.round(
        [np.min(nonGround_array[:, 0]), np.max(nonGround_array[:, 0])], 3
    )
    yLimits = np.round(
        [np.min(nonGround_array[:, 1]), np.max(nonGround_array[:, 1])], 3
    )

    # Ground raster in x-y-layer
    xRaster = np.round(
        np.arange(xLimits[0], xLimits[1] + step, step), 3
    )  # arange excludes stop-value -> + step
    yRaster = np.round(np.arange(yLimits[0], yLimits[1] + step, step), 3)

    # initialize vector with zeros to save z-coords
    z_coords = np.zeros((np.size(x_y_coords, axis=0), 1))
    # iterate through all x- and y-values and get the fitting z-value from the ground grid
    for i, (x, y) in enumerate(x_y_coords):
        # search for x-value in xRaster to get the index, which indicates the fitting z-value
        # location in the ground grid
        idx_x = np.searchsorted(xRaster, x)
        # same for y
        idx_y = np.searchsorted(yRaster, y)
        z_coords[i, 0] = ground_grid[idx_x, idx_y]

    # return all three coords together [x, y, z]
    coords = np.append(x_y_coords, z_coords, axis=1)
    return coords
