import numpy as np
import itertools
import subprocess
import pathlib
import os
import datetime

import data_loading


def create_cube_off_file(
        center_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
        side_len: float = 1.0,
        output_path: str = "cube.off") -> np.ndarray[np.float32]:
    """Returns the coordinates of the cube's vertices"""
    # writing fixed information to the file
    fp = open(output_path, "w+")
    num_vertices = 8
    num_faces = 6
    num_edges = 12
    fp.write("OFF\n")
    fp.write(f"{num_vertices} {num_faces} {num_edges}\n")

    center_arr = np.array(center_point)
    halves = (-side_len/2, side_len/2)
    offsets = np.array(list(itertools.product(halves, repeat=3)))
    vertex_coordinates = np.array([center_arr+offset for offset in offsets])
    for point in vertex_coordinates:
        fp.write(" ".join([str(coord) for coord in point])+"\n")
    # due to how itertools.prudct orders its elements, we can know beforehand
    # which vertices will belong to which faces
    # there are two faces with equal xs, two with equal ys, two with equal zs
    # the order of vertices within faces matters!
    equal_xs = [[0, 1, 3, 2], [4, 5, 7, 6]]
    equal_ys = [[0, 1, 5, 4], [2, 3, 7, 6]]
    equal_zs = [[0, 2, 6, 4], [1, 3, 7, 5]]
    faces = equal_xs+equal_ys+equal_zs

    for face in faces:
        # 4 because every face in a cube has 4 vertices
        fp.write(f"4 {' '.join([str(vertex) for vertex in face])}\n")
    fp.close()
    return vertex_coordinates


def scale_from_coords(vertex_coords: np.ndarray[np.float32]) -> float:
    min_val = np.min(vertex_coords)
    max_val = np.max(vertex_coords)
    max_offset = max_val
    if np.abs(min_val) > max_val:
        max_offset = np.abs(min_val)
    # it could be made a bit bigger, but this is a safe option
    return 1 / (4 * max_offset)


def scale_from_off(off_path: str) -> float:
    # discard the tuple with model stats and the array containing faces
    _, vertex_coords, _ = data_loading.load_off_file_np(off_path)
    return scale_from_coords(vertex_coords)


def create_and_render_cube(
        center_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
        side_len: float = 1.0,
        cube_filepath: str = "new_cube.off",
        rendered_pics_path: str = "./cube_pics/") -> None:
    """The function expects to be run from the top-level repository directory
    (the one which contains .gitignore)"""
    vertex_coords = create_cube_off_file(center_point, side_len, cube_filepath)
    scale = scale_from_coords(vertex_coords)
    arguments = ["blender", "--background", "--python", "./src/render_blender.py", "--",
                f"{cube_filepath}", "--output_folder", f"{rendered_pics_path}",
                "--scale", f"{scale}"]
    subprocess.run(arguments)
    print(scale)


def render_offs_from_pattern(
        glob_pattern: str,
        dataset_root: str = "ModelNet40",
        dest_dir: str = "RGBD_ModelNet40") -> None:
    path_obj = pathlib.Path(dataset_root).glob(glob_pattern)
    out_log = open("/dev/null", "w")
    for i, filename in enumerate(path_obj):
        out_dir = str(os.path.dirname(filename)).lstrip(dataset_root)
        out_dir = f"{dest_dir}{out_dir}/"
        # skip if the corresponding png already exsists
        first_out_file = os.path.basename(filename).split('.')[0]
        if os.path.isfile(f"{out_dir}{first_out_file}_r_000.png"):
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] SKIP {filename}: {i+1}")
            continue
        scale = scale_from_off(str(filename))
        arguments = ["blender", "--background", "--python", "./src/render_blender.py", "--",
                f"{filename}", "--output_folder", f"{out_dir}",
                "--scale", f"{scale}"]
        subprocess.run(arguments, stdout=out_log)
        # there are 12311 files in the main dataset
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {filename}: {i+1}")
    out_log.close()


def export_centered_off(src_off_path: str, dest_off_path: str) -> None:
    params, vertices, faces = data_loading.load_off_file_np(src_off_path)
    centered_vertices = vertices - np.mean(vertices, axis=0)
    data_loading.export_off_file_np(params, centered_vertices, faces, dest_off_path)


def create_centered_off_dataset(
        glob_pattern: str,
        dataset_root: str = "ModelNet40",
        dest_dir: str = "ModelNet40_centered") -> None:
    """Parses all files in `dataset_root` according to glob pattern.
    Recreates the dataset's structure in `dest_dir`, but all with models in
    .off files shifted to have their centers of mass at (0,0,0). Translates
    all vertices by a vector composed of the average values of x, y, z"""
    path_obj = pathlib.Path(dataset_root).glob(glob_pattern)
    for i, src_file_path in enumerate(path_obj):
        out_dir = str(os.path.dirname(src_file_path)).lstrip(dataset_root)
        out_dir = f"{dest_dir}{out_dir}/"
        # skip if the corresponding .off if it already exsists
        out_file_name = os.path.basename(src_file_path).split('.')[0]
        if os.path.isfile(f"{out_dir}{out_file_name}.off"):
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] SKIP {src_file_path}: {i+1}")
            continue
        # print(f"{out_dir}{out_file_name}")
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        export_centered_off(src_file_path, f"{out_dir}{out_file_name}.off")
        # there are 12311 files in the main dataset
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {src_file_path}: {i+1}")


def render_off_dataset_set_numbers(
        n_train: int = 64, n_test: int = 20,
        dataset_root: str = "ModelNet40_centered",
        dest_dir: str = "RGBD_ModelNet40_centered") -> None:
    # no glob, because I want to be ABSOLUTELY SURE that the same order is always preserved
    dataset_root_dir = pathlib.Path(dataset_root)
    out_log = open("/dev/null", "w")
    for class_dir in dataset_root_dir.iterdir():
        class_name = os.path.basename(class_dir)
        train_files = sorted(list(pathlib.Path(f"{class_dir}/train").iterdir()))
        out_dir_train = f"{dest_dir}/{class_name}/train/"
        for train_off_file in train_files[:n_train]:
            out_file_name = os.path.basename(train_off_file).split('.')[0]
            if os.path.isfile(f"{out_dir_train}{out_file_name}_r_000.png"):
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] SKIP train/{out_file_name}")
                continue
            scale = scale_from_off(str(train_off_file))
            arguments = ["blender", "--background", "--python", "./src/render_blender.py", "--",
                f"{train_off_file}", "--output_folder", f"{out_dir_train}",
                "--scale", f"{scale}"]
            subprocess.run(arguments, stdout=out_log)
        out_dir_test = f"{dest_dir}/{class_name}/test/"
        test_files = sorted(list(pathlib.Path(f"{class_dir}/test").iterdir()))
        for test_off_file in test_files[:n_test]:
            out_file_name = os.path.basename(test_off_file).split('.')[0]
            if os.path.isfile(f"{out_dir_test}{out_file_name}_r_000.png"):
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] SKIP test/{out_file_name}")
                continue
            scale = scale_from_off(str(test_off_file))
            arguments = ["blender", "--background", "--python", "./src/render_blender.py", "--",
                f"{test_off_file}", "--output_folder", f"{out_dir_test}",
                "--scale", f"{scale}"]
            subprocess.run(arguments, stdout=out_log)


if __name__ == "__main__":
    # create_and_render_cube(side_len=1.0)
    # render_offs_from_pattern("*/*/*.off")
    # create_centered_off_dataset("*/*/*.off")
    render_off_dataset_set_numbers()
