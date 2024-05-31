import open3d as o3d
import numpy as np

import datetime
import os
import pathlib

from voxelization import export_binary_voxel_grid

def voxelize_images(rgb_image_path):
    # Load images as o3d Image
    rgb_image = o3d.io.read_image(str(rgb_image_path))
    d_image = o3d.io.read_image(str(rgb_image_path).rstrip(".png")+"_depth0001.png")
    # Combine images into o3d RGBD Image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image, d_image, depth_scale=1.0, depth_trunc=3.0, convert_rgb_to_intensity=True)

    # Create camera parameters based on the Blender camera
    params = o3d.camera.PinholeCameraIntrinsic(fx=280, fy=280, cx=128, cy=128, width=256, height=256)

    try:
        # Create point cloud from RGBD image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            params,
            project_valid_depth_only=False
            )

        # Remove points with depth 1.0 (these are the background points)
        points = np.asarray(pcd.points)
        mask = points[:,2] != 1.0
        pcd.points = o3d.utility.Vector3dVector(points[mask])
        # o3d.visualization.draw_geometries([pcd])

        # Voxelization
        bb = pcd.get_axis_aligned_bounding_box()
        size = max(bb.get_extent())/63

        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=size)

        voxels = voxel_grid.get_voxels()
        indices = np.array([vx.grid_index for vx in voxels])
        indices = np.array([vx.grid_index for vx in voxels])
        grid = np.zeros((64, 64, 64), dtype=np.uint8)
        grid[indices[:,0], indices[:,1], indices[:,2]] = 1
    except:
        print(f"Error in {rgb_image_path}")
        grid = np.zeros((64, 64, 64), dtype=np.uint8)

    return grid

def create_voxelized_dataset_set_numbers_from_rgbd(
        n_train: int = 64, n_test: int = 20, n_voxels: int = 64,
        dataset_root: str = "RGBD_ModelNet40_centered",
        dest_dir: str = "ModelNet40_voxel_input") -> None:
    """Creates a voxelized version of the RGBD images.
    Each file stores the positions of bits set to 1 within
    a flattened version of the voxel."""
    # no glob, because I want to be ABSOLUTELY SURE that the same order is always preserved
    dataset_root_dir = pathlib.Path(dataset_root)
    for class_dir in dataset_root_dir.iterdir():
        class_name = os.path.basename(class_dir)
        train_files = sorted(list(pathlib.Path(f"{class_dir}/train").iterdir()))
        out_dir_train = f"{dest_dir}/{class_name}/train/"
        for i, train_file in enumerate(train_files):
            if "depth" in str(train_file):
                continue
            out_file_name = os.path.basename(train_file).split('.')[0]
            if os.path.isfile(f"{out_dir_train}{out_file_name}.txt"):
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] SKIP train/{out_file_name}")
                continue
            voxel_grid = voxelize_images(f"./{train_file}")
            pathlib.Path(out_dir_train).mkdir(parents=True, exist_ok=True)
            export_binary_voxel_grid(voxel_grid, f"{out_dir_train}{out_file_name}.txt")
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {train_file}: {i+1}")
        out_dir_test = f"{dest_dir}/{class_name}/test/"
        test_files = sorted(list(pathlib.Path(f"{class_dir}/test").iterdir()))
        for i, test_file in enumerate(test_files):
            if "depth" in str(test_file):
                continue
            out_file_name = os.path.basename(test_file).split('.')[0]
            if os.path.isfile(f"{out_dir_test}{out_file_name}.txt"):
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] SKIP test/{out_file_name}")
                continue
            voxel_grid = voxelize_images(f"./{test_file}")
            pathlib.Path(out_dir_test).mkdir(parents=True, exist_ok=True)
            export_binary_voxel_grid(voxel_grid, f"{out_dir_test}{out_file_name}.txt")
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {test_file}: {i+1}")

if __name__ == "__main__":
    create_voxelized_dataset_set_numbers_from_rgbd()