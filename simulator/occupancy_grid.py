import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

# from bresenhan import bresenhamline


def _bresenhamline_nslope(slope):
    """
    Normalize slope for Bresenham's line algorithm.

    >>> s = np.array([[-2, -2, -2, 0]])
    >>> _bresenhamline_nslope(s)
    array([[-1., -1., -1.,  0.]])

    >>> s = np.array([[0, 0, 0, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  0.,  0.]])

    >>> s = np.array([[0, 0, 9, 0]])
    >>> _bresenhamline_nslope(s)
    array([[ 0.,  0.,  1.,  0.]])
    """
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope


def _bresenhamlines(start, end, max_iter):
    """
    Returns npts lines of length max_iter each. (npts x max_iter x dimension)

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> _bresenhamlines(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[[ 3,  1,  8,  0],
            [ 2,  1,  7,  0],
            [ 2,  1,  6,  0],
            [ 2,  1,  5,  0],
            [ 1,  0,  4,  0],
            [ 1,  0,  3,  0],
            [ 1,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0]],
    <BLANKLINE>
           [[ 0,  0,  2,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0, -2,  0],
            [ 0,  0, -3,  0],
            [ 0,  0, -4,  0],
            [ 0,  0, -5,  0],
            [ 0,  0, -6,  0]]])
    """
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    # Approximate to nearest int
    return np.array(np.rint(bline), dtype=start.dtype)


def bresenhamline(start, end, max_iter=5):
    """
    Returns a list of points from (start, end] by ray tracing a line b/w the
    points.
    Parameters:
        start: An array of start points (number of points x dimension)
        end:   An end points (1 x dimension)
            or An array of end point corresponding to each start point
                (number of points x dimension)
        max_iter: Max points to traverse. if -1, maximum number of required
                  points are traversed

    Returns:
        linevox (n x dimension) A cumulative array of all points traversed by
        all the lines so far.

    >>> s = np.array([[3, 1, 9, 0],[0, 0, 3, 0]])
    >>> bresenhamline(s, np.zeros(s.shape[1]), max_iter=-1)
    array([[ 3,  1,  8,  0],
           [ 2,  1,  7,  0],
           [ 2,  1,  6,  0],
           [ 2,  1,  5,  0],
           [ 1,  0,  4,  0],
           [ 1,  0,  3,  0],
           [ 1,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0,  2,  0],
           [ 0,  0,  1,  0],
           [ 0,  0,  0,  0],
           [ 0,  0, -1,  0],
           [ 0,  0, -2,  0],
           [ 0,  0, -3,  0],
           [ 0,  0, -4,  0],
           [ 0,  0, -5,  0],
           [ 0,  0, -6,  0]])
    """
    # Return the points as a single array
    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])


class VoxelGrid:
    def __init__(self, grid_size=500, grid_resolution=0.1, occupancy=False, K=None):
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.voxel_grid = None
        self.check_occupancy = occupancy
        if self.check_occupancy:
            self.occupancy_grid = np.ones((self.grid_size, self.grid_size)) * -1

        # intrinsics
        self.hfov = np.pi / 2
        if K is None:
            self.K = np.array(
                [
                    [1 / np.tan(self.hfov / 2.0), 0.0, 0.0, 0.0],
                    [0.0, 1 / np.tan(self.hfov / 2.0), 0.0, 0.0],
                    [0.0, 0.0, 1, 0],
                    [0.0, 0.0, 0, 1],
                ]
            )
        else:
            self.K = K

        self.initialized = False

    def insert_pointcloud(self, ptcloud, pose):
        # transform to world frame
        rotation_matrix = Rotation.from_quat(pose[3:]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = pose[:3]
        ptcloud_t = (
            T @ np.concatenate((ptcloud, np.ones((ptcloud.shape[0], 1))), axis=1).T
        ).T[:, :3]

        # update voxel grid
        pcd = o3d.geometry.PointCloud()
        if self.voxel_grid is not None:
            ptcloud_grid = np.asarray(
                [
                    self.voxel_grid.origin
                    + self.grid_resolution / 2.0
                    + pt.grid_index * self.voxel_grid.voxel_size
                    for pt in self.voxel_grid.get_voxels()
                ]
            )
            pcd.points = o3d.utility.Vector3dVector(
                np.concatenate((ptcloud_grid, ptcloud_t), axis=0)
            )
        else:
            pcd.points = o3d.utility.Vector3dVector(ptcloud_t)
        self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, self.grid_resolution
        )

        # update occupany grid
        if self.check_occupancy:
            self.update_occupancy(ptcloud_t, pose)

    def get_voxel_grid(self):
        voxel_grid_np = np.zeros(
            (
                int(self.grid_size / self.grid_resolution),
                int(self.grid_size / self.grid_resolution),
                int(self.grid_size / self.grid_resolution),
            )
        )
        # get voxel centers as points
        point_cloud_grid = np.asarray(
            [
                self.voxel_grid.origin
                + self.grid_resolution / 2.0
                + pt.grid_index * self.voxel_grid.voxel_size
                for pt in self.voxel_grid.get_voxels()
            ]
        )
        # point_cloud_grid = np.asarray([self.voxel_grid.get_voxel_center_coordinate(pt.grid_index) for pt in self.voxel_grid.get_voxels()])
        # set occupancy in voxel grid
        for pt in point_cloud_grid:
            i = np.min([int(pt[0] / self.grid_resolution), voxel_grid_np.shape[0] - 1])
            j = np.min([int(pt[1] / self.grid_resolution), voxel_grid_np.shape[2] - 1])
            k = np.min([int(pt[2] / self.grid_resolution), voxel_grid_np.shape[1] - 1])
            voxel_grid_np[i, k, j] = 1

        return voxel_grid_np

    def pointcloud_from_depth(self, depth):
        # Create a mask of valid depths (not NaNs)
        valid_depth_mask = ~np.isnan(depth)

        H, W = depth.shape

        xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))

        # Apply the mask to depth, xs, and ys
        valid_depth = depth[valid_depth_mask]
        valid_xs = xs[valid_depth_mask]
        valid_ys = ys[valid_depth_mask]

        # Unproject
        xys = np.vstack(
            (
                valid_xs * valid_depth,
                valid_ys * valid_depth,
                -valid_depth,
                np.ones(valid_depth.shape),
            )
        )
        xy = np.matmul(np.linalg.inv(self.K), xys)
        xy = xy.T[:, :3]

        return xy

    def insert_depth_image(self, depth_img, pose):
        ptcloud = self.pointcloud_from_depth(depth_img)
        if ptcloud.shape[0] > 0:
            self.insert_pointcloud(ptcloud, pose)
            self.initialized = True
            return True
        return False

    def get_pointcloud(self):
        ptcloud_grid = np.asarray(
            [
                self.voxel_grid.origin
                + self.grid_resolution / 2.0
                + pt.grid_index * self.voxel_grid.voxel_size
                for pt in self.voxel_grid.get_voxels()
            ]
        )
        return ptcloud_grid

    def viz_voxel_grid(self):
        ptcloud_grid = np.asarray(
            [
                self.voxel_grid.origin
                + self.grid_resolution / 2.0
                + pt.grid_index * self.voxel_grid.voxel_size
                for pt in self.voxel_grid.get_voxels()
            ]
        )
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ptcloud_grid)
        o3d.visualization.draw_geometries([pcd])

    def get_occupancy_grid(self):
        return self.occupancy_grid

    def update_occupancy(self, ptcloud, pose):
        # filter points by height
        ptcloud = ptcloud[ptcloud[:, 1] > 1]
        ptcloud = ptcloud[ptcloud[:, 1] < 2]

        # flatten ptcloud
        grid2d = np.zeros((ptcloud.shape[0], 2))
        grid2d[:, 0] = ptcloud[:, 0]
        grid2d[:, 1] = ptcloud[:, 2]

        # raycast for occupancy
        for pt in grid2d:
            start_pt = np.array(
                [[pose[0] / self.grid_resolution, pose[2] / self.grid_resolution]]
            )

            # adjust to 0 centered grid
            start_pt = start_pt + np.array([[self.grid_size / 2, self.grid_size / 2]])
            end_pt = np.array(
                [
                    [
                        int(pt[0] / self.grid_resolution + self.grid_size / 2),
                        int(pt[1] / self.grid_resolution) + self.grid_size / 2,
                    ]
                ]
            )
            bresenham_path = bresenhamline(start_pt, end_pt, max_iter=-1)

            # set as free
            for bres_pt in bresenham_path:
                x = int(bres_pt[0])
                y = int(bres_pt[1])
                self.occupancy_grid[x, y] = 0
            x = int(pt[0] / self.grid_resolution + self.grid_size / 2)
            y = int(pt[1] / self.grid_resolution + self.grid_size / 2)

            # set as occupied
            self.occupancy_grid[x, y] = 1

        # set current position as free
        x = int(pose[0] / self.grid_resolution + self.grid_size / 2)
        y = int(pose[2] / self.grid_resolution + self.grid_size / 2)
        self.occupancy_grid[x, y] = 0
