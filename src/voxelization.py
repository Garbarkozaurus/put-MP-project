import open3d as o3d
import numpy as np

import datetime
import os
import pathlib
import torch


def voxelize(
        path: str, visualize: bool = False,
        n_voxels: int = 64) -> np.ndarray[np.uint8]:
    """Creates a voxel grid with specified number of voxels per axis.
    Centers the objects, returns a numpy array with a 1 if voxel is full,
    or 0 if it is empty.
    """
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
    grid = np.zeros((n_voxels, n_voxels, n_voxels), dtype=np.uint8)
    grid[indices[:,0], indices[:,1], indices[:,2]] = 1
    return grid

def visualize_voxel_grid(voxel_grid: o3d.geometry.VoxelGrid) -> None:
    o3d.visualization.draw_geometries([voxel_grid])


def export_binary_voxel_grid(
        grid: np.ndarray[np.uint8],
        export_path: str) -> None:
    with open(export_path, "w+") as fp:
        ones_indices = np.flatnonzero(grid)
        for index in ones_indices:
            fp.write(f"{index}\n")


def load_ones_indices_into_binary_grid(
        indices_filepath: str,
        grid_shape: tuple[int, int, int] = (64, 64, 64)
        ) -> np.ndarray[np.uint8]:
    """Loads a file which stores indices that are to be set to 1, each within
    a separate line. The remaining positions within the array of shape
    `grid_shape` will be set to 0."""
    fp = open(indices_filepath, "r")
    positions = [int(line) for line in fp.readlines()]
    fp.close()
    grid = np.zeros(np.prod(grid_shape), dtype=np.uint8)
    grid[positions] = 1
    return grid.reshape(grid_shape)


def create_voxel_from_ones_indices(
        ones_indices: np.ndarray[np.uint8],
        grid_shape: tuple[int, int, int] = (64, 64, 64)
        ) -> o3d.geometry.VoxelGrid:
    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.voxel_size = 1
    for one_idx in ones_indices:
        voxel = o3d.geometry.Voxel()
        voxel.color = np.array([0, 0, 0])
        # grid_idx = np.array([one_idx % grid_shape[2], one_idx // grid_shape[1], one_idx // (grid_shape[1] * grid_shape[2])])
        grid_idx = np.array([one_idx // (grid_shape[0] * grid_shape[1]), (one_idx // grid_shape[1]) % grid_shape[2], one_idx % grid_shape[2]])
        print(grid_idx)
        voxel.grid_index = grid_idx
        voxel_grid.add_voxel(voxel)
    return voxel_grid


def create_voxel_from_binary_grid(
        binary_grid: np.ndarray[np.uint8]
        ) -> o3d.geometry.VoxelGrid:
    voxel_grid = o3d.geometry.VoxelGrid()
    voxel_grid.voxel_size = 1
    for z in range(binary_grid.shape[0]):
        for y in range(binary_grid.shape[1]):
            for x in range(binary_grid.shape[2]):
                if binary_grid[x, y, z] == 0:
                    continue
                voxel = o3d.geometry.Voxel()
                voxel.color = np.array([0, 0, 0])
                voxel.grid_index = np.array([x, y, z])
                voxel_grid.add_voxel(voxel)
    return voxel_grid


def voxel_from_predictions(
        prediction_tensor: torch.tensor,
        binarization_threshold: float = 0.5) -> o3d.geometry.VoxelGrid:
    binary_inputs = torch.where(prediction_tensor >= binarization_threshold, 1, 0)
    voxel = create_voxel_from_binary_grid(binary_inputs)
    return voxel


def create_voxelized_dataset_set_numbers(
        n_train: int = 64, n_test: int = 20, n_voxels: int = 64,
        dataset_root: str = "ModelNet40_centered",
        dest_dir: str = "ModelNet40_ones") -> None:
    """Creates a voxelized version of the .off dataset.
    Each file stores the positions of bits set to 1 within
    a flattened version of the voxel."""
    # no glob, because I want to be ABSOLUTELY SURE that the same order is always preserved
    dataset_root_dir = pathlib.Path(dataset_root)
    for class_dir in dataset_root_dir.iterdir():
        class_name = os.path.basename(class_dir)
        train_files = sorted(list(pathlib.Path(f"{class_dir}/train").iterdir()))
        out_dir_train = f"{dest_dir}/{class_name}/train/"
        for i, train_off_file in enumerate(train_files[:n_train]):
            out_file_name = os.path.basename(train_off_file).split('.')[0]
            if os.path.isfile(f"{out_dir_train}{out_file_name}.txt"):
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] SKIP train/{out_file_name}")
                continue
            voxel_grid = voxelize(f"./{train_off_file}", n_voxels)
            pathlib.Path(out_dir_train).mkdir(parents=True, exist_ok=True)
            export_binary_voxel_grid(voxel_grid, f"{out_dir_train}{out_file_name}.txt")
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {train_off_file}: {i+1}")
        out_dir_test = f"{dest_dir}/{class_name}/test/"
        test_files = sorted(list(pathlib.Path(f"{class_dir}/test").iterdir()))
        for i, test_off_file in enumerate(test_files[:n_test]):
            out_file_name = os.path.basename(test_off_file).split('.')[0]
            if os.path.isfile(f"{out_dir_test}{out_file_name}.txt"):
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] SKIP test/{out_file_name}")
                continue
            voxel_grid = voxelize(f"./{test_off_file}", n_voxels)
            pathlib.Path(out_dir_test).mkdir(parents=True, exist_ok=True)
            export_binary_voxel_grid(voxel_grid, f"{out_dir_test}{out_file_name}.txt")
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {test_off_file}: {i+1}")


if __name__ == "__main__":
    # grid = voxelize("./ModelNet40/airplane/train/airplane_0003.off", False, 64)
    # print("Grid stats:")
    # print("Shape:", grid.shape)
    # print("Max:", np.max(grid))
    # print("Min:", np.min(grid))
    # print("Mean:", np.mean(grid))
    # EXPORT_PATH = "test_grid.txt"
    # export_binary_voxel_grid(grid, EXPORT_PATH)
    # bench, chair, couch, table
    INPUT_PATH = "ModelNet40_voxel_input/bench/test/bench_0174_r_120.txt"
    OUTPUT_PATH = "ModelNet40_ones/bench/test/bench_0174.txt"
    imported_in = load_ones_indices_into_binary_grid(INPUT_PATH)
    imported_out = load_ones_indices_into_binary_grid(OUTPUT_PATH)
    voxel_in = create_voxel_from_binary_grid(imported_in)
    voxel_out = create_voxel_from_binary_grid(imported_out)
    o3d.visualization.draw_geometries([voxel_out])
    o3d.visualization.draw_geometries([voxel_in])

    # imported = load_ones_indices_into_binary_grid(EXPORT_PATH)
    # print("Shape:", imported.shape)
    # print("Max:", np.max(imported))
    # print("Min:", np.min(imported))
    # print("Mean:", np.mean(imported))
    # indices = [int(x) for x in open(EXPORT_PATH).readlines()]
    # voxel_grid = create_voxel_from_ones_indices(indices)
    # o3d.visualization.draw_geometries([voxel_grid])
    # voxel_grid = create_voxel_from_binary_grid(imported)
    # o3d.visualization.draw_geometries([voxel_grid])
    # create_voxelized_dataset_set_numbers()
