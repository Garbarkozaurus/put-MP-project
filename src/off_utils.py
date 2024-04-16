import numpy as np
import itertools
import subprocess

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
    # discard the array containing faces
    vertex_coords, _ = data_loading.load_off_file_np(off_path)
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


if __name__ == "__main__":
    create_and_render_cube(side_len=1.0)