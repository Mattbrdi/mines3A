#
#
#      0===========================0
#      |    MAREVA 3D Modelling    |
#      0===========================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Script of the practical session. Plane detection by RANSAC
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 19/09/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"

import numpy as np

# Import functions to read and write ply files
from utils.ply import write_ply, read_ply

# Import time package
import time


# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points):
    point = points[0]
    x = points[1] - points[0]
    y = points[2] - points[0]
    normal = np.cross(x, y)
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        raise ValueError('The three points do not define a plane')
    normal = normal / normal_norm

    return point, normal


def in_plane(points, ref_pt, normal, threshold_in=0.1):
    
    indices = np.zeros(len(points), dtype=bool)
    
    # TODO: return a boolean mask of points in range
    return np.abs(np.dot((points - ref_pt), normal)) < threshold_in

    # for i in range(len(points)): 
    #     x = points[i] - ref_pt 
    #     eps = np.abs(x@normal) 
    #     indices[i] = eps < threshold_in
        
    # return indices


def RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1):
    if len(points) < 3 or NB_RANDOM_DRAWS < 1 or threshold_in <= 0:
        raise ValueError('RANSAC needs at least three points, one draw and a positive threshold')

    best_ref_pt = None
    best_normal = None
    best_vote = 0
    rng = np.random.default_rng()
    for k in range(NB_RANDOM_DRAWS):
        choosen_points = rng.choice(points, size=3, replace=False)
        try:
            ref_pt, normal = compute_plane(choosen_points)
        except ValueError:
            continue
        vote = in_plane(points, ref_pt, normal, threshold_in).sum()

        if vote > best_vote:
            best_normal = normal
            best_ref_pt = ref_pt
            best_vote = vote

    if best_ref_pt is None:
        raise ValueError('No valid plane found')
    return best_ref_pt, best_normal


def multi_RANSAC(points, NB_RANDOM_DRAWS=100, threshold_in=0.1, NB_PLANES=2):
    """Extract planes, returning original indices and a label per extracted point."""
    if NB_RANDOM_DRAWS < 1 or threshold_in <= 0:
        raise ValueError('RANSAC needs at least one draw and a positive threshold')

    plane_inds = np.empty(0, dtype=int)
    remaining_inds = np.arange(len(points))
    plane_labels = np.empty(0, dtype=int)

    for plane_label in range(NB_PLANES):
        if len(remaining_inds) < 3:
            break

        remaining_points = points[remaining_inds]
        try:
            ref_pt, normal = RANSAC(remaining_points, NB_RANDOM_DRAWS, threshold_in)
        except ValueError:
            # No non-degenerate plane was found among the remaining points.
            break

        mask = in_plane(remaining_points, ref_pt, normal, threshold_in)
        if mask.sum() < 3:
            break

        # Map the local mask back to indices in the original cloud.
        selected_inds = remaining_inds[mask]
        plane_inds = np.concatenate((plane_inds, selected_inds))
        plane_labels = np.concatenate((plane_labels,
                                       np.full(len(selected_inds), plane_label, dtype=int)))
        remaining_inds = remaining_inds[~mask]

    return plane_inds, remaining_inds, plane_labels


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    N = len(points)

    # Computes the plane passing through 3 randomly chosen points
    # ***********************************************************
    #

    if True:

        # Define parameter
        threshold_in = 0.1

        # Take randomly three points
        pts = points[np.random.randint(0, N, size=3)]

        # Computes the plane passing through the 3 points
        t0 = time.time()
        ref_pt, normal = compute_plane(pts)
        t1 = time.time()
        print('plane computation done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        t0 = time.time()
        points_in_plane = in_plane(points, ref_pt, normal, threshold_in)
        t1 = time.time()
        print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
        plane_inds = points_in_plane.nonzero()[0]

        # Save the 3 points and their corresponding plane for verification
        pts_clr = np.zeros_like(pts)
        pts_clr[:, 0] = 1.0
        write_ply('../triplet.ply',
                  [pts, pts_clr],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply('../triplet_plane.ply',
                  [points[plane_inds], colors[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Computes the best plane fitting the point cloud
    # ***********************************
    #

    if True:

        # Define parameters of RANSAC
        NB_RANDOM_DRAWS = 100
        threshold_in = 0.05

        # Find best plane by RANSAC
        t0 = time.time()
        best_ref_pt, best_normal = RANSAC(points, NB_RANDOM_DRAWS, threshold_in)
        t1 = time.time()
        print('RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Find points in the plane and others
        points_in_plane = in_plane(points, best_ref_pt, best_normal, threshold_in)
        plane_inds = points_in_plane.nonzero()[0]
        remaining_inds = (1-points_in_plane).nonzero()[0]

        # Save the best extracted plane and remaining points
        write_ply('../best_plane.ply',
                  [points[plane_inds], colors[plane_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])
        write_ply('../remaining_points.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

    # Find multiple planes in the cloud
    # *********************************
    #

    if True:

        # Define parameters of multi_RANSAC
        NB_RANDOM_DRAWS = 200
        threshold_in = 0.05
        NB_PLANES = 5

        # Recursively find best plane by RANSAC
        t0 = time.time()
        plane_inds, remaining_inds, plane_labels = multi_RANSAC(points, NB_RANDOM_DRAWS, threshold_in, NB_PLANES)
        t1 = time.time()
        print('\nmulti RANSAC done in {:.3f} seconds'.format(t1 - t0))

        # Save the best planes and remaining points
        write_ply('../best_planes.ply',
                  [points[plane_inds], colors[plane_inds], plane_labels.astype(np.int32)],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'plane_label'])
        write_ply('../remaining_points_.ply',
                  [points[remaining_inds], colors[remaining_inds]],
                  ['x', 'y', 'z', 'red', 'green', 'blue'])

        print('Done')
