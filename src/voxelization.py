import open3d as o3d
import numpy as np

def voxelize(path, visualize=False, n_voxels=64):
    '''
    Creates a voxel grid with specified number of voxels per axis. Centers the objects, returns a numpy array with a 1 if voxel is full, or 0 if it is empty.
    '''
    mesh = o3d.io.read_triangle_mesh(path)

    # fit to unit cube
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
            center=mesh.get_center())
    if visualize:
        o3d.visualization.draw_geometries([mesh])

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=1.0/(n_voxels-1))
    if visualize:
        o3d.visualization.draw_geometries([voxel_grid])

    voxels = voxel_grid.get_voxels()
    indices = np.array([vx.grid_index for vx in voxels])
    grid = np.zeros((n_voxels, n_voxels, n_voxels))
    grid[indices[:,0], indices[:,1], indices[:,2]] = 1.0

    return grid

if __name__ == "__main__":
    grid = voxelize("./ModelNet40/airplane/train/airplane_0003.off", True, 64)
    print("Grid stats:")
    print("Shape:", grid.shape)
    print("Max:", np.max(grid))
    print("Min:", np.min(grid))
    print("Mean:",np.mean(grid))