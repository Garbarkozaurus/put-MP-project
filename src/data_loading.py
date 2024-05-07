import numpy as np
import pathlib
from os import rename, remove


def load_off_file_np(
        file_path: str) -> tuple[
                        tuple[int, int, int],
                        np.ndarray[np.float32],
                        np.ndarray[np.int32]]:
    """The return tuple has three elements: a three-int tuple,
    and two np.ndarrays. The tuple stores the number of vertices,
    faces and edges in the model (in this order - as they are given
    in the file). The first of the matrices contains vertex coordinates.
    The second - which vertex belongs to which face"""
    with open(file_path, "r") as file:
        if 'OFF' != file.readline().strip():
            raise ValueError('Not a valid OFF header')
        n_verts, n_faces, n_edges = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for _ in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for _ in range(n_faces)]
    return (n_verts, n_faces, n_edges), np.array(verts, dtype=np.float32), np.array(faces, np.int32)


def export_off_file_np(
        model_params: tuple[int, int, int],
        vertices: np.ndarray[np.float32],
        faces: np.ndarray[np.int32],
        file_path: str) -> None:
    fp = open(file_path, "w+")
    fp.write("OFF\n")
    fp.write(" ".join(map(str, model_params))+"\n")
    for vertex in vertices:
        fp.write(" ".join(map(str, vertex))+"\n")
    for face in faces:
        fp.write(f"{len(face)} {' '.join(map(str, face))}\n")


def iterate_over_off_models(
        glob_pattern: str,
        dataset_root: str = "./ModelNet40/") -> None:
    """Example glob patterns:
    '*/*/*.off' - get all .off files
    '*/train/*.off' - get all .off files designated to teh trianing set
    'airplane/test/*.off' - tel all .off files from testing for class airplane
    """
    path_obj = pathlib.Path(dataset_root).glob(glob_pattern)
    for file in path_obj:
        print(file)
        load_off_file_np(file)


def fix_invalid_files(glob_pattern: str, dataset_root: str = "./ModelNet40/") -> None:
    """Fixes the files with first line of pattern: 'OFFn_verts n_faces n_edges'
    by replacing them with a file of the same name, and the first line split
    into two: 'OFF' and 'n_verts n_faces n_edges' """
    path_obj = pathlib.Path(dataset_root).glob(glob_pattern)
    for file_name in path_obj:
        fp = open(file_name, "r")
        first_line = fp.readline().strip()
        if first_line == "OFF":
            continue
        if first_line[:3] == "OFF":
            new_first_line = "OFF"
            new_second_line = first_line[3:]
            # n_verts, n_faces, n_edges = tuple([int(x) for x in new_second_line.split()])
        new_file = open(f"{file_name}_temp", "w+")
        new_file.write(f"{new_first_line}\n")
        new_file.write(f"{new_second_line}\n")
        while True:
            next_line = fp.readline()
            if not next_line:
                break
            new_file.write(next_line)
        fp.close()
        new_file.close()
        remove(file_name)
        rename(f"{file_name}_temp", file_name)


if __name__ == "__main__":
    fix_invalid_files("*/test/*.off")
    iterate_over_off_models("*/test/*.off")
