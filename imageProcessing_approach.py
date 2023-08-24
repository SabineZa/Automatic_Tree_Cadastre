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

# %%

from math import pi
from pathlib import Path
import numpy as np
from tabulate import tabulate as tab  # https://pypi.org/project/tabulate/
import csv
import cv2  # opencv-contrib-python -> https://pypi.org/project/opencv-python/
import time

# scikit-image for circular hough transform and for region properties
from skimage import color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.measure import label, regionprops_table

import ground as gr

# %%


def imageProcessing_approach(
    nonGround_array: np.ndarray,
    ground_grid: np.ndarray,
    pxlsize_binary_img,
    pxlsize_crown_img,
    num_peaks: int,
    save: bool = False,
    visualize: bool = False,
    ground_rastersize: float = 1.0,
) -> None:
    """Extract tree parameters (tree position, tree height, dbh and tree crown diameter)
    from a point cloud using images.

    Parameters
    ----------
    nonGround_array : np.ndarray
        Point cloud without ground. (x, y, z)
    ground_grid : np.ndarray
        Raster grid containing the median of the z-values of all points in each raster cell. [x,y]
    pxlsize_binary_img : _type_
        Pixelsize of the binary images
    pxlsize_crown_img : _type_
        Pixelsize of the crown image
    num_peaks : int
        Maximum number of peaks in each Hough space
    save : bool, optional
        If True images of interim results are saved, by default False
    visualize : bool, optional
        If True some interim results and the result image are visualized, by default False
    ground_rastersize : float, optional
        Size of a ground raster cell, by default 1.0
    """

    # 1  *********** CREATE IMAGES *****************************************************
    start = time.time()

    # 1.1  Create greyscale image of whole point cloud
    height_img, grayscale_img = createImage_max(nonGround_array, pxlsize_crown_img)
    if visualize:
        cv2.imshow("Grayscale Image", grayscale_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    saveImage(grayscale_img, "grayscale_img.png", save)

    # Median and Closing
    # openCV Morphological Transformations:
    # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    # https://www.geeksforgeeks.org/python-opencv-morphological-operations/
    grayscale_img_fltrd = cv2.medianBlur(grayscale_img, 5)
    if visualize:
        cv2.imshow("Median", grayscale_img_fltrd)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    kernel = np.array(
        [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]], np.uint8
    )
    grayscale_img_fltrd = cv2.morphologyEx(grayscale_img_fltrd, cv2.MORPH_CLOSE, kernel)
    if visualize:
        cv2.imshow("Closing", grayscale_img_fltrd)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    saveImage(grayscale_img_fltrd, "grayscale_img_filtered.png", save)

    # normalize values to get grayscale image with values ranged from 0 to 1
    grayscale_img_norm = (grayscale_img_fltrd - np.min(grayscale_img_fltrd)) / (
        np.max(grayscale_img_fltrd) - np.min(grayscale_img_fltrd)
    )

    # 1.2  Create binary images
    # Get two point cloud layers
    # z-value range 1.2 to 1.4
    dbh_layer = nonGround_array[1.2 < nonGround_array[:, 2]]
    dbh_layer = dbh_layer[1.4 > dbh_layer[:, 2]]
    # z-value range 0.4 to 1.4
    trunk_layer = nonGround_array[0.4 < nonGround_array[:, 2]]
    trunk_layer = trunk_layer[trunk_layer[:, 2] < 1.4]

    # Get binary images of the two point cloud layers
    binary_dbh = createImage_his(dbh_layer, nonGround_array, pxlsize_binary_img)
    binary_trunk = createImage_his(trunk_layer, nonGround_array, pxlsize_binary_img)
    saveImage(binary_dbh, "binary_dbh.png", save)
    saveImage(binary_trunk, "binary_trunk.png", save)

    end = time.time()
    print("It took {:5.3f}s to create all three images.\n".format(end - start), end=" ")

    # 2  ******** PROCESS BINARY IMAGES TO DETECT TREES ****************************************
    start = time.time()

    # 2.1  binary_dbh image
    binary_dbh2, circles_dbh, centers_dbh, radii_dbh = find_trees(
        binary_dbh, pxlsize_binary_img, num_peaks
    )
    saveImage(binary_dbh2, "binary_dbh2.png", save)
    saveImage(circles_dbh, "circles_dbh.png", save)

    # 2.2  binary_trunk image
    binary_trunk2, circles_trunk, centers_trunk, radii_trunk = find_trees(
        binary_trunk, pxlsize_binary_img, num_peaks
    )
    saveImage(binary_trunk2, "binary_trunk2.png", save)
    saveImage(circles_trunk, "circles_trunk.png", save)

    # 2.3  Find circles which are similar in both images -> trunk_img validates dbh_img

    centers, radii = find_similarCircles(
        centers_dbh, centers_trunk, radii_dbh, radii_trunk, 3
    )
    # Draw circles
    rgb_dbh2 = color.gray2rgb(binary_dbh2)
    for center_y, center_x, radius in zip(centers[:, 0], centers[:, 1], radii):
        circy, circx = circle_perimeter(
            center_y, center_x, radius, shape=rgb_dbh2.shape
        )
        rgb_dbh2[circy, circx] = (0, 255, 0)
    saveImage(rgb_dbh2, "found_similiarCircles_dbh2.png", save)
    print(f"{len(radii)} similar circles found in circles_dbh and circles_trunk.")

    # Convert from pixelsize trunk dbh image to pixelsize tree crown image
    centers = centers / (pxlsize_crown_img / pxlsize_binary_img)
    radii = radii / (pxlsize_crown_img / pxlsize_binary_img)

    end = time.time()
    print("It took {:5.3f}s to detect trees.\n".format(end - start), end=" ")

    # 3  ********* FINAL TREE PARAMETERS ********************************************
    start = time.time()

    # 3.1  Mean tree crown diameter and tree height

    # Find a tree crown for every tree trunk with region growing
    tree_h = np.empty((len(centers), 1))
    mean_crown_diameter = np.empty((len(centers), 1))
    treecrown_props = {}

    for i in range(len(centers)):
        # get crown as binary image
        _, mask = regionGrowing(
            grayscale_img_norm,
            np.around(centers[i, :]).astype(np.uint32),
            0.1,
            150,
            15,
            True,
        )

        # Find properties of tree crown
        # https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html
        label_mask = label(mask)
        treecrown_props[i] = regionprops_table(
            label_mask,
            grayscale_img_norm,
            properties=(
                "area",
                "axis_major_length",
                "axis_minor_length",
                "eccentricity",
                "orientation",
                "centroid_weighted",
            ),
        )

        # Calculate tree height
        tree_h[i, 0] = determine_tree_height(height_img, mask)

        # Calculate mean crown diameter [m]
        axis_major_length = (
            treecrown_props[i]["axis_major_length"][0] * pxlsize_crown_img
        )  # [m]
        axis_minor_length = (
            treecrown_props[i]["axis_minor_length"][0] * pxlsize_crown_img
        )  # [m]
        mean_crown_diameter[i, 0] = np.around(
            (axis_major_length + axis_minor_length) / 2
        )  # [m]

    # 3.2  DBH

    # Calculate diameter at breast height (DBH)
    dbh = (radii * 2) * pxlsize_crown_img  # [m]
    dbh = np.reshape(dbh, (len(dbh), 1))

    # 3.3  Tree Positions

    # convert tree positions (image coordinates) back to the original
    # coordinate system of the input point cloud -> (x, y) [m]
    tree_coords2D = convert_coordinates(
        centers, grayscale_img_norm, nonGround_array, pxlsize_crown_img
    )
    # add z-coordinate to get 3D-position of the tree -> (x, y, z) [m]
    tree_coords = gr.transform_back_z(
        tree_coords2D, ground_grid, nonGround_array, ground_rastersize
    )

    # all tree params in one matrix
    tree_params = np.concatenate(
        (tree_coords, centers, tree_h, dbh, mean_crown_diameter), axis=1
    )

    # 3.4  Sort out trees
    tree_params = sort_out_trees(tree_params)
    # tree_params = coords (x, y, z), centers (cy, cx), tree_h, dbh, mean_crown_diameter
    tree_coords = tree_params[:, 0:3]
    centers = tree_params[:, 3:5]
    tree_h = tree_params[:, 5]
    dbh = tree_params[:, 6]
    mean_crown_diameter = tree_params[:, 7]
    # tree_params_result = coords (x, y, z), tree_h, dbh, mean_crown_diameter
    # no image coords (centers)
    tree_params_result = np.delete(tree_params, [3, 4], 1)

    end = time.time()
    print(
        "It took {:5.3f}s to get final tree parameters.\n".format(end - start), end=" "
    )

    # 4  ********************** VISUALIZE AND SAVE RESULTS *********************************
    start = time.time()

    # 4.1  Create and save image with numbered tree positions and tree crown extents
    # https://www.geeksforgeeks.org/python-opencv-cv2-circle-method/
    img = cv2.imread("Images/grayscale_img_filtered.png")
    window_name = "Result Image"
    tree_idxs = np.arange(len(centers)).astype(str)

    for i, c in enumerate(centers):
        c = np.around(c).astype(np.uint64)
        # swap columns
        c[[0, 1]] = c[[1, 0]]
        # tree center point
        img = cv2.circle(img, c, 4, (0, 170, 255), -1)

        # crown circle
        img = cv2.circle(
            img,
            c,
            np.around(mean_crown_diameter[i] / 2 * pxlsize_crown_img * 100).astype(
                np.uint64
            ),
            (0, 255, 0),
            2,
        )

        # add number to tree
        # https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
        img = cv2.putText(
            img,
            tree_idxs[i],
            c + 5,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    saveImage(img, "tree_result.png", save, "output/")
    if visualize:
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 4.2  Write tree params in table and latex
    # https://pypi.org/project/tabulate/
    headers = [
        "x [m]",
        "y [m]",
        "z [m]",
        "Baumhöhe [m]",
        "DBH [m]",
        "Kronendurchmesser [m]",
    ]
    print(
        tab(
            tree_params_result,
            headers,
            tablefmt="fancy_grid",
            floatfmt=(".0f", ".3f", ".3f", ".3f", ".0f", ".2f", ".0f"),
            showindex="always",
        )
    )
    print(
        tab(
            tree_params_result,
            headers,
            tablefmt="latex",
            floatfmt=(".0f", ".3f", ".3f", ".3f", ".0f", ".2f", ".0f"),
            showindex="always",
        )
    )

    # 4.3  Write tree params to csv
    p = Path("output/")
    p.mkdir(parents=True, exist_ok=True)

    with open(
        p / "treeParams.csv",
        "w",
        newline="",
    ) as csvfile:
        csvwriter = csv.writer(
            csvfile, delimiter=";", quotechar="|", quoting=csv.QUOTE_MINIMAL
        )
        csvwriter.writerow(headers)
        csvwriter.writerows(tree_params_result)

    end = time.time()
    print(
        "It took {:5.3f}s to visualize and save results.".format(end - start), end=" "
    )


# ************************ END OF IPA FUNCTION *****************************


# %% 1  CREATE IMAGES


def createImage_max(
    pointcloud_array: np.ndarray, pixelsize: np.float64
) -> tuple[np.ndarray, np.ndarray]:
    """Creates a 2D grayscale image [y,x] from the input pointcloud (x,y,z).
    Thereby the z-axis is projected into a plane in a raster.

    Parameters
    ----------
    pointcloud_array : np.ndarray
        Contains the 3D point cloud in a numpy-array
    pixelsize : np.float64
        Size of one pixel in the image in [m].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Contains a height image [y, x] with the actual height
        value in each cell and a 2D grayscale image [y,x]
        with a value range from 0 to 255 with dtype('uint8')
        (compatible for openCV) representing the height.
    """

    pixelsize = pixelsize * 100  # [cm]
    # Limits of the image defined by pcd size
    xLimits = [np.min(pointcloud_array[:, 0]), np.max(pointcloud_array[:, 0])]
    yLimits = [np.min(pointcloud_array[:, 1]), np.max(pointcloud_array[:, 1])]
    # Initialize image with NaN-values
    height_img = np.full(
        (
            (
                np.around((yLimits[1] - yLimits[0]) * pixelsize).astype(np.int64) + 1,
                np.around((xLimits[1] - xLimits[0]) * pixelsize).astype(np.int64) + 1,
            )
        ),
        np.nan,
    )

    for i in range(len(pointcloud_array) - 1):
        # Determine the grid cell from the coordinates
        x = (
            np.around(pointcloud_array[i, 0] * pixelsize)
            - np.around(xLimits[0] * pixelsize)
        ).astype(np.int64)
        y = (
            np.around(pointcloud_array[i, 1] * pixelsize)
            - np.around(yLimits[0] * pixelsize)
        ).astype(np.int64)
        # If NaN or a smaller z-value is stored in the image cell then store new value
        if height_img[y, x] < pointcloud_array[i, 2] or np.isnan(height_img[y, x]):
            height_img[y, x] = pointcloud_array[i, 2]

    # Convert image matrix to grayscale image usable with openCV
    height_img = np.flip(height_img, axis=0)
    # normalize values to get grayscale image with values ranged from 0 to 255
    img_max = np.max(height_img[~np.isnan(height_img)])
    img_min = np.min(height_img[~np.isnan(height_img)])
    grayscale_img = np.around(
        ((height_img - img_min) / (img_max - img_min)) * 255, 0
    ).astype(np.uint8)
    # white background instead of black
    grayscale_img[grayscale_img == 0] = 255

    return height_img, grayscale_img


def createImage_his(
    pointcloud_array: np.ndarray, pcd_defineSize: np.ndarray, rastersize: np.float64
) -> np.ndarray:
    """Creates a 2D binary image from the input point cloud
    based on the associated histogram in a grid.

    Parameters
    ----------
    pointcloud_array : np.ndarray
        Contains the 3D point cloud in a numpy-array
    pcd_defineSize : np.ndarray
        Point cloud that defines limits of the resulting image. Useful to generate several images
        with the same size but different pointcloud_arrays.
    rastersize : float
        Pixel size of the image raster [m]

    Returns
    -------
    np.ndarray
       Contains the 2D binary image with dtype('uint8')
        -> compatible for openCV
    """

    # Limits of the image
    xLimits = [np.min(pcd_defineSize[:, 0]), np.max(pcd_defineSize[:, 0])]
    yLimits = [np.min(pcd_defineSize[:, 1]), np.max(pcd_defineSize[:, 1])]

    # Initialize image with zeros
    bins_x = np.round(
        np.arange(round(xLimits[0], 1), round(xLimits[1], 1), rastersize), 3
    )
    bins_y = np.round(
        np.arange(round(yLimits[0], 1), round(yLimits[1], 1), rastersize), 3
    )
    img = np.zeros((len(bins_y), len(bins_x)), np.uint8)

    # i represents the x-value of one rastercell
    for counterX, i in enumerate(bins_x):
        # segment all x values in the current raster cell i
        dx = pointcloud_array[pointcloud_array[:, 0] > i, :]
        dx = dx[dx[:, 0] < i + rastersize, :]
        # compute histogram for this raster cell
        yhis, _ = np.histogram(
            dx[:, 1],
            bins=len(bins_y),
            range=(np.min(bins_y).astype(float), np.max(bins_y).astype(float)),
        )
        img[:, counterX] = np.flip(yhis, axis=0)

    # compute binary image based on histogram
    # threshold = 0.5 -> one point is enough to cause a white pixel
    _, binary_img = cv2.threshold(img, 0.5, 255, cv2.THRESH_BINARY)

    return binary_img


# %% 2  PROCESS BINARY IMAGES TO DETECT TREES


def find_trees(
    binary_img: np.ndarray, pixelsize: np.float64, num_peaks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Function to find trees in image. Uses bwareafilt() to remove unwanted contours before
    detecting circles with hough-transform.

    Parameters
    ----------
    binary_img : np.ndarray
        Binary image
    pixelsize : np.float64
        Pixelsize of the image
    num_peaks : int
        Maximum number of peaks in each Hough space

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Returns adapted binary image, binary image with drawn circles,
        centers and radii of found circles.
    """

    # Closing with a large kernel to connect unwanted contours / clusters and display tree trunks
    # as round as possible
    kernel = np.ones((6, 6), np.uint8)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    # Remove unwanted contours / clusters
    # Binary image is clustered and clusters with more or less pixels than specified are removed
    # Wert 1 m aus Leitfaden zu Stadtbäumen in Bayern. Handlungsempfehlungen aus dem Projekt
    # Stadtbäume – Wachstum, Umweltleistungen und Klimawandel. Zentrum Stadtnatur und Klimaanpassung
    max_dbh = 1  # [m]
    upper_limit = (pi * max_dbh**2) / (pixelsize**2 * 4)
    binary_img2, _ = bwareafilt(binary_img, area_range=(1, upper_limit))
    binary_img2 = binary_img2.astype(np.uint8) * 255

    # Find circles in images
    # Opening to remove noise and get the tree trunks as round as possible
    kernel = np.ones((5, 5), np.uint8)
    binary_img2 = cv2.morphologyEx(binary_img2, cv2.MORPH_OPEN, kernel)
    circles_img, centers, radii = houghTransform_skimage(
        binary_img2, pixelsize, num_peaks
    )

    return binary_img2, circles_img, centers, radii


# %%


def bwareafilt(
    mask: np.ndarray, area_range: tuple = (0, np.inf)
) -> tuple[np.ndarray, np.ndarray]:
    """Extract objects from binary image by size.
    Equivalent to bwareafilt() in Matlab. Source:
    https://github.com/AndersDHenriksen/SanityChecker/blob/master/AllChecks.py
    With own adaptations (n eliminated).

    Parameters
    ----------
    mask : np.ndarray
        Binary image
    area_range : tuple, optional
        Clusters with the number of pixels in the specified range are kept, by default (0, np.inf)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A binary image containing only those objects that meet the criteria and the kept areas
    """

    _, labels = cv2.connectedComponents(mask.astype(np.uint8))
    area_idx = np.arange(1, np.max(labels) + 1)
    areas = np.array([np.sum(labels == i) for i in area_idx])
    inside_range_idx = np.logical_and(areas >= area_range[0], areas <= area_range[1])
    area_idx = area_idx[inside_range_idx]
    areas = areas[inside_range_idx]
    keep_idx = area_idx[np.argsort(areas)[::-1]]
    kept_areas = areas[np.argsort(areas)[::-1]]
    if np.size(kept_areas) == 0:
        kept_areas = np.array([0])

    kept_mask = np.isin(labels, keep_idx)

    return kept_mask, kept_areas


# %%


def houghTransform_skimage(
    image: np.ndarray, pxlsize: np.float64, num_peaks: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform a circular Hough transform with scikit-image

    Parameters
    ----------
    image : np.ndarray
        Grayscale image
    pxlsize : np.float64
        Pixelsize of the image [m]
    num_peaks : int
        Maximum number of peaks in each Hough space

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Resulting rgb image, centers of the circles and radii
    """

    edges = canny(image, sigma=2)
    # sigma -> width of the Gaussian filter (the noisier the image, the greater the width)
    # low and high threshold for the hysteresis thresholding
    # https://scikit-image.org/docs/stable/auto_examples/edges/plot_canny.html

    # Perform a circular Hough transform
    hough_radii = np.arange(
        1, np.around(1 / pxlsize).astype(np.int64), 1
    )  # Maximum radius to detect 1 meter

    hough_res = hough_circle(edges, hough_radii)

    # Return peaks in a circle Hough transform
    _, cx, cy, radii = hough_circle_peaks(
        hough_res,
        hough_radii,
        min_xdistance=1,
        min_ydistance=1,
        num_peaks=num_peaks,
        # total_num_peaks=100,
        normalize=True,
    )
    print(f"{len(radii)} circles detected by circular Hough transform.")

    # Draw them
    image = color.gray2rgb(image)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius, shape=image.shape)
        image[circy, circx] = (0, 255, 0)

    cx = np.reshape(cx, (len(cx), 1))
    cy = np.reshape(cy, (len(cy), 1))
    centers = np.concatenate((cy, cx), axis=1)

    return image, centers, radii


# %%


def find_similarCircles(
    centers1: np.ndarray,
    centers2: np.ndarray,
    radii1: np.ndarray,
    radii2: np.ndarray,
    tol: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find similar circles by comparing the properties (center points and radii) of circles, which
    were found by the circular Hough transform in two binary images. For the actual comparison
    the function ismembertol is used.

    Parameters
    ----------
    centers1 : np.ndarray
        Center coordinates (cy, cx) of the circles of the first binary image.
    centers2 : np.ndarray
        Center coordinates (cy, cx) of the circles of the second binary image.
    radii1 : np.ndarray
        Radii of the circles of the first binary image.
    radii2 : np.ndarray
        Radii of the circles of the second binary image.
    tol : int
        Tolerance in pixel in which the elements should match.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Center points (cy, cx) and radii of the circles, which occur in both images.
    """

    radii1 = np.reshape(radii1, (len(radii1), 1))
    radii2 = np.reshape(radii2, (len(radii2), 1))
    circles1 = np.concatenate((centers1, radii1), axis=1)
    circles2 = np.concatenate((centers2, radii2), axis=1)

    ismember = ismembertol(circles1, circles2, tol)

    # Keep only the rows of the array with the entries of the circles that appear in both images
    circles1 = np.concatenate((circles1, ismember), axis=1)
    circles1 = circles1[circles1[:, 3] == 1, :]

    centers1 = circles1[:, 0:2]
    radii1 = circles1[:, 2]

    return centers1, radii1


def ismembertol(array1: np.ndarray, array2: np.ndarray, tol: int) -> np.ndarray:
    """Programmed as equivalent of the Matlab function ismembertol.
    Returns an array containing True, where the elements of array1 in one row are
    within tolerance of the elements in array2 line by line.

    Parameters
    ----------
    array1 : np.ndarray
        First input array which is compared to second input array.
    array2 : np.ndarray
        Second input array.
    tol : int
        Tolerance in pixel in which the elements should match.

    Returns
    -------
    np.ndarray
        Logical array containing True wherever the elements in one row in array1 are
        members of array2 within tolerance.
    """

    ismember_matrix = np.full((np.shape(array1)[0], 1), fill_value=False)
    for r1 in range(np.shape(array1)[0]):
        for r2 in range(np.shape(array2)[0]):
            if np.all(np.abs(array1[r1, :] - array2[r2, :]) <= tol):
                ismember_matrix[r1, 0] = True
                break

    return ismember_matrix


# %% 3  FINAL TREE PARAMETERS


def regionGrowing(
    img: np.ndarray,
    initPos: np.ndarray,
    thresh: float,
    maxDist: int,
    maxDistMitte: int,
    tfFillHoles: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Region grwoing algorithm for 2D grayscale images with range 0 to 1.

    Copyright (c) 2011, Daniel
    All rights reserved.
    Modeled after the Matlab function regionGrowing() programmed by Daniel Kellner (2011)
    with own adaptations. License in Repository included.

    Parameters
    ----------
    img : np.ndarray
        2D grayscale image (rows, cols)
    initPos : np.ndarray
        Coordinates for initial seed position.
    thresh : float
        Absolute threshold level to be included.
    maxDist : int
        Maximum distance to the initial position [px].
    maxDist2 : int
        Second maximum distance to the initial position [px].
    tfFillHoles : bool
        If true, small enclosed holes in the binary mask are closed with kernel(5,5).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Contours of the enclosing polygon. Each contour is stored as a vector of points.
        Binary mask with the same shape as input image indicating 1 for associated pixel
        and 0 for outside.
    """

    nRow = np.shape(img)[0]
    nCol = np.shape(img)[1]
    # initial pixel value
    regVal = img[initPos[0], initPos[1]]
    # preallocate array
    mask = np.full((nRow, nCol), False)
    # add initial pixel to the queue
    queue = np.array([[initPos[0], initPos[1]]])

    # *********** Start of region growing algorithm
    while len(queue) > 0:
        # the first queue position determines the new values
        xv = queue[0, 0]
        yv = queue[0, 1]
        # Set regVal to current value
        regVal = img[xv, yv]
        # delete first queue pos
        queue = np.delete(queue, 0, 0)

        # check the neighbors for the current pos
        # range() stop value not included -> neighbors -1 and 1; current pixel 0
        for i in range(-1, 2):
            for j in range(-1, 2):

                if (
                    xv + i > 0
                    and xv + i < nRow  # within the x-bounds?
                    and yv + j > 0
                    and yv + j < nCol  # within the y-bounds?
                    and np.any([i, j])  # i/j of (0/0) is redundant!
                    and not mask[xv + i, yv + j]  # pixelposition already set?
                    and np.sqrt((xv + i - initPos[0]) ** 2 + (yv + j - initPos[1]) ** 2)
                    < maxDistMitte  # within distance?
                    and img[xv + i, yv + j]
                    >= (regVal - thresh)  # within range of the threshold?
                    and img[xv + i, yv + j] != 1
                ):
                    # current pixel is true, if all properties are fulfilled
                    mask[xv + i, yv + j] = True
                    # add the current pixel to the computation queue (recursive)
                    queue = np.append(queue, [[xv + i, yv + j]], axis=0)

                elif (
                    xv + i > 0
                    and xv + i < nRow  # within the x-bounds?
                    and yv + j > 0
                    and yv + j < nCol  # within the y-bounds?
                    and np.any([i, j])  # i/j of (0/0) is redundant!
                    and not mask[xv + i, yv + j]  # pixelposition already set?
                    and np.sqrt((xv + i - initPos[0]) ** 2 + (yv + j - initPos[1]) ** 2)
                    < maxDist  # within distance?
                    and img[xv + i, yv + j]
                    <= (regVal + thresh / 100)  # within range of the threshold?
                    and img[xv + i, yv + j] >= (regVal - thresh)
                ):
                    # current pixel is true, if all properties are fulfilled
                    mask[xv + i, yv + j] = True
                    # add the current pixel to the computation queue (recursive)
                    queue = np.append(queue, [[xv + i, yv + j]], axis=0)

    # **************** End of region growing algorithm

    # fill holes and extract the polygon vertices
    # extract the enclosing polygon
    if tfFillHoles:
        # fill the holes inside the mask
        # In Matlab, the imfill() function is used, which corresponds not exactly with
        # the closing function used here. Result fulfills its purpose.
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        # https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # swap x- and y-values
    contours[0][:, :, [1, 0]] = contours[0][:, :, [0, 1]]

    return contours, mask


# %%


def determine_tree_height(height_img: np.ndarray, mask: np.ndarray) -> np.float16:
    """Determines tree height out of a height image
    with the help of a tree crown mask. The pixel of the height image
    within the tree canopy mask with the highest value results in the tree height.

    Parameters
    ----------
    height_img : np.ndarray
        Height image with actual height values.
    mask : np.ndarray
        Binary mask which indicates tree crown.

    Returns
    -------
    np.float128
        Tree height
    """
    height_img = np.nan_to_num(height_img)
    crown_heights = np.multiply(height_img, mask)
    tree_height = np.max(crown_heights)

    return tree_height


# %%


def convert_coordinates(
    coords_img: np.ndarray,
    img_array: np.ndarray,
    pointcloud_array: np.ndarray,
    pixelsize: np.float64,
) -> np.ndarray:
    """Convert image coordinates [y,x] back to the coordinate system of the input point cloud.
    The origin of the image coordinate system is in the upper left corner, while the origin of
    the original (and target) coordinate system of the point cloud is in the lower left corner.
    Used formulas:
    xp [m] = xi [pxl] * pxlsize [m] + min_xp [m]
    yp [m] = (max_yi [pxl] - yi [pxl]) * pxlsize [m] + min_yp [m]

    Parameters
    ----------
    coords_img : np.ndarray
        Points (y, x) in the image coordinate system [pixel]
    img_array : np.ndarray
        Image which represents the image coordinate system [pixel]
    pointcloud_array : np.ndarray
        Point cloud with points (x, y) in the target coordinate system. [m]
    pixelsize : np.float64
        Size of one pixel in the image [m]

    Returns
    -------
    np.ndarray
        Former image points now in the coordinate system of the input point cloud (x, y) [m]
    """

    # Minima of the point cloud correspond to the lower left corner of the image
    min_p = [
        np.min(pointcloud_array[:, 0]),
        np.min(pointcloud_array[:, 1]),
    ]  # [m]

    # xp [m] = xi [pxl] * pxlsize [m] + min_xp [m]
    xp = np.reshape(coords_img[:, 1] * pixelsize + min_p[0], (len(coords_img), 1))
    # yp [m] = (max_yi [pxl] - yi [pxl]) * pxlsize [m] + min_yp [m]
    yp = np.reshape(
        (np.shape(img_array)[0] - coords_img[:, 0]) * pixelsize + min_p[1],
        (len(coords_img), 1),
    )
    coords = np.concatenate((xp, yp), axis=1)

    return coords


# %%


def sort_out_trees(tree_params: np.ndarray) -> np.ndarray:
    """All trees that do not meet certain criteria are sorted out.
    No tree if ...
    ... tree height, dbh or mean crown diameter is negative
    ... the tree is too close to another tree
    ... the dbh ist larger than the mean crown diameter
    ... the dbh is larger than the tree height
    ... the tree height is smaller than the mean crown diameter.

    Parameters
    ----------
    tree_params : np.ndarray
        Tree parameters [m] containing: coords (x, y, z), centers (cy, cx), tree_h, dbh, mean_crown_diameter

    Returns
    -------
    np.ndarray
        Tree parameters [m] containing: coords (x, y, z), centers (cy, cx), tree_h, dbh, mean_crown_diameter

    """
    # No tree if ...
    del_idx = np.empty((0, 0), dtype=np.int64)
    for i, (x, y, z, cy, cx, tree_h, dbh, mean_crown_d) in enumerate(tree_params):
        # ... tree height, dbh or mean crown diameter is negative
        if tree_h < 0 or dbh < 0 or mean_crown_d < 0:
            del_idx = np.append(del_idx, i)
            print(
                f"Tree number {i} was deleted because tree height, dbh or mean crown diameter is negative."
            )
        elif np.any(
            np.abs(calc_dist_to_trees(x, y, z, tree_params[i + 1 :, 0:3])) < dbh
        ):
            # ... the tree is too close to another tree
            del_idx = np.append(del_idx, i)
            print(
                f"Tree number {i} was deleted because it is too close to another tree (duplicated tree)."
            )
        elif dbh > mean_crown_d:
            # ... the dbh ist larger than the mean crown diameter.
            del_idx = np.append(del_idx, i)
            print(
                f"Tree number {i} was deleted because dbh is larger than the mean crown diameter."
            )
        elif dbh > tree_h:
            # ... the dbh is larger than the tree height
            del_idx = np.append(del_idx, i)
            print(
                f"Tree number {i} was deleted because dbh is larger than the tree height."
            )
        elif tree_h < mean_crown_d:
            # ... the tree height is smaller than the mean crown diameter.
            del_idx = np.append(del_idx, i)
            print(
                f"Tree number {i} was deleted because the tree height is smaller than the mean crown diameter."
            )

    tree_params = np.delete(tree_params, del_idx, 0)
    return tree_params


def calc_dist_to_trees(
    tree_base_x, tree_base_y, tree_base_z, tree_coords: np.ndarray
) -> np.ndarray:
    """Calculation of the respective distances between one tree to all other trees.

    Parameters
    ----------
    tree_base_x : Any
        x-coordinate of the tree base point [m]
    tree_base_y : Any
        y-coordinate of the tree base point [m]
    tree_base_z : Any
        z-coordinate of the tree base point [m]
    tree_coords : np.ndarray
        Coordinates (x, y, z) [m] of the other trees to which the distance should be calculated.

    Returns
    -------
    np.ndarray
        Array containing the distances [m] from the input tree to the trees in the input array.
    """
    return np.sqrt(
        np.power(tree_base_x - tree_coords[:, 0], 2)
        + np.power(tree_base_y - tree_coords[:, 1], 2)
        + np.power(tree_base_z - tree_coords[:, 2], 2)
    )


#  %%  HELPER FUNCTIONS


def saveImage(
    image: np.ndarray, filename: str, save: bool, filepath: str = "Images/"
) -> None:
    """Save image as file

    Parameters
    ----------
    image : np.ndarray
        Image which should be saved
    filepath : str
        Filepath to the destination folder (optional). Default: "Images/"
    filename : str
        Filename of the image in the destination folder
    save : bool
        Whether the image should be saved or not
    """

    p = Path(filepath)
    p.mkdir(parents=True, exist_ok=True)

    if save:
        isWritten = cv2.imwrite(str(p / filename), image)
        if isWritten:
            print(f"Image {filename} is successfully saved as file to {filepath}.")
        else:
            print(f"Error while saving image {filename} as file.")
