import typing
from torch.utils.data import Dataset
from os import listdir
from os.path import isfile, isdir, join
from enum import Enum
from typing import Dict, List, Tuple
import torch
import time
import pyexr
from os.path import join, splitext
import json
from torch.tensor import Tensor


class BufferType(Enum):
    ALBEDO = 1
    DI = 2
    WS_NORMALS = 3
    CS_NORMALS = 4
    CS_POSITIONS = 5
    DEPTH = 6
    GT_RTGI = 7


BUFFER_TYPE_TO_NAME = {
    BufferType.ALBEDO: "albedo",
    BufferType.DI: "di",
    BufferType.WS_NORMALS: "ws_normals",
    BufferType.CS_NORMALS: "cs_normals",
    BufferType.CS_POSITIONS: "cs_positions",
    BufferType.DEPTH: "depth",
    BufferType.GT_RTGI: "rtgi",
}

SIZE_KEY = "size"
INPUT_BUFFERS_KEY = "input_buffers"
RESOLUTION_KEY = "resolution"
DYNAMIC_RANGE_KEY = "dynamic_range"

# from https://stackoverflow.com/questions/24481852/serialising-an-enum-member-to-json
class BufferTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BufferType):
            return {"__enum__": str(obj)}
        return json.JSONEncoder.default(self, obj)


class GIDataset(Dataset):
    """
    Each of the Train/Test/Validate datasets has similar hierarchical structure.
    - The first level of directories in the parent directory denotes the 3D scene that were used
        to obtain the images - [Auditorium, bamboo_house, bathroom, ...].
    - For each 3D scene directory we have child dirs for the different types of images obtained:
        [albedo, rtgi, di, ws_normals, cs_normals, cs_positions, depth]. The ground truth images are
        in the rtgi dir and are obtained by real-time ray tracing. Input images are di (direct
        illumination), ws_normals, cs_normals (normals in world and camera space),
        depth (in clip space), cs_positions (in clip space).
    - For each image type we have child dirs for the dynamic range - [hdr, ldr]. HDR images are
        in linear color scale and saved in .EXR file format. They look dimmer when viewer with
        an external viewer and need additional mapping to sRGB for viewing purposes. The ldr images
        are in non-linear sRGB color scale and are saved as .PNG files.
    - For each dynamic range dir we have resolution dirs as childred - [256, 512]. Each image is
        in square shape. The children of resolution directories are the individual images for the chosen
        3D scene -> type -> dynamic range -> resolution.
    """

    def __init__(
        self,
        root_dir: str,
        input_buffers: List[BufferType],
        useHDR: bool = True,
        resolution: int = 512,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir

        self.dynamic_range = "hdr" if useHDR else "ldr"

        if resolution not in [256, 512]:
            raise Exception("Illegal resolution value! Expected 256 or 512 but got " + str(resolution))
        self.resolution = str(resolution)

        if BufferType.GT_RTGI in input_buffers:
            raise Exception("Ground truth is part of the input but not expected!")
        self.input_buffers = input_buffers

        self.scene_dirs: List[str] = self.get_scene_dirs()

        num_images_per_scene: List[int] = self.get_scene_dirs_sizes(self.scene_dirs)
        self.directory_lookup_list: List[int] = self.accumulated_sizes_per_scene(num_images_per_scene)
        # scene_path -> [filename]
        self.gt_filenames_cache: Dict[str, int] = self.get_filenames_cache(self.scene_dirs)

        self.validate_dataset()

    def __len__(self) -> int:
        return self.calculate_dataset_size()

    def __getitem__(self, idx: int) -> torch.Tensor:
        s1 = time.time_ns()
        (scene_index, image_index) = self.get_dataset_and_image_index(idx)
        scene_path = self.scene_dirs[scene_index]

        image_name = self.gt_filenames_cache[scene_path][image_index]
        # print("Preparing indices: {} ms".format((time.time_ns() - s1) / 1_000_000.0))

        tensors_list = []
        for input_buffer in self.input_buffers:
            s_i = time.time_ns()
            file = pyexr.open(
                join(scene_path, BUFFER_TYPE_TO_NAME[input_buffer], self.dynamic_range, self.resolution, image_name)
            )
            s_i_1 = time.time_ns()
            # Cut out the unused alpha channel. In the case of depth images take just one channel.
            # All the non-alpha channels are equal there.
            image_array = file.get()
            s_i_2 = time.time_ns()

            image_array = image_array[:, :, :1] if input_buffer == BufferType.DEPTH else image_array[:, :, :3]
            # print(
            #     "Loading buffer {} total: {} ms, file load: {} ms, file.get(): {} ms".format(
            #         input_buffer,
            #         ((time.time_ns() - s_i) / 1_000_000.0),
            #         (s_i_1 - s_i) / 1_000_000.0,
            #         (s_i_2 - s_i_1) / 1_000_000.0
            #     )
            # )

            tensors_list.append(torch.from_numpy(image_array))

        s2 = time.time_ns()

        concatenated = torch.cat(tensors_list, 2)
        # [512w, 512h, num_channels] -> [num_channels, 512w, 512h]
        concatenated = concatenated.permute(2, 0, 1)
        # print("Concatenating tensors: {} ms".format((time.time_ns() - s2) / 1_000_000.0))

        # torch.save(concatenated, "tensor.pt")
        # s3 = time.time_ns()
        # asd = torch.load("tensor.pt")
        # print("Loading tensor from memory: {} ms".format((time.time_ns() - s3) / 1_000_000.0))

        # print("Total: {} ms".format((time.time_ns() - s1) / 1_000_000.0))

        return concatenated

    def transform_and_save_dataset_as_tensors(self, target_dir: str):
        """Loading the data images (especially the EXR files) and transforming them to numpy arrays
        and pytorch tensors is quite slow. With such schema loading a single batch for 32 files
        (32 x num_buffer images) can take roughly a second for the full set of buffers. Thus we cache
        the dataset as pre-concatenated tensors for the given resolution and dynamic range. Then the
        ready-to-use cached version of the dataset can be used with GIDatasetCached for a substantial
        speedup.
        This method should be used sparingly only as a pre-processing step and the results should be
        reused for a number of network runs. The whole dataset is quite big (>110 GB) and this process
        can rapidly lower the write cycles life of a SSD.
        """

        total_size = self.calculate_dataset_size()

        dataset_descr = {
            SIZE_KEY: total_size,
            INPUT_BUFFERS_KEY: self.input_buffers,
            RESOLUTION_KEY: self.resolution,
            DYNAMIC_RANGE_KEY: self.dynamic_range,
        }

        with open(join(target_dir, "dataset_descr.json"), "w") as fp:
            json.dump(dataset_descr, fp, cls=BufferTypeEncoder)
            print("Saved dataset description.")

        for i in range(total_size):
            (scene_index, image_index) = self.get_dataset_and_image_index(i)
            scene_path = self.scene_dirs[scene_index]
            image_name = self.gt_filenames_cache[scene_path][image_index]

            tensors_list = []
            for input_buffer in self.input_buffers:
                file = pyexr.open(
                    join(scene_path, BUFFER_TYPE_TO_NAME[input_buffer], self.dynamic_range, self.resolution, image_name)
                )
                # Cut out the unused alpha channel. In the case of depth images take just one channel.
                # All the non-alpha channels are equal for depth images.
                image_array = file.get()
                image_array = image_array[:, :, :1] if input_buffer == BufferType.DEPTH else image_array[:, :, :3]

                tensors_list.append(torch.from_numpy(image_array))

            # [512w, 512h, num_channels] -> [num_channels, 512w, 512h]
            concatenated: Tensor = torch.cat(tensors_list, 2).permute(2, 0, 1)
            no_ext_name = splitext(image_name)[0]
            output_name = "{}.pt".format(no_ext_name)
            output_path = join(target_dir, output_name)
            torch.save(concatenated, output_path)
            print("Transformed and saved {}".format(output_path))

    def get_filenames_cache(self, scene_dirs: List[str]) -> Dict[str, int]:
        """We cache all the gt file names so we don't have to iterate the dataset folders each time"""
        result: Dict[str, int] = {}
        for scene_dir in scene_dirs:
            gt_path = join(scene_dir, BUFFER_TYPE_TO_NAME[BufferType.GT_RTGI], self.dynamic_range, self.resolution)
            result[scene_dir] = self.all_filepaths_for_files_in_path(gt_path)

        return result

    def get_dataset_and_image_index(self, idx: int) -> Tuple[int, int]:
        """
        We use the accumulated positions as a lookup to speed up finding the index of the scene which corresponds to idx
        and the image index in that scene
        """
        dataset_index = 0
        accumulated = 0
        for (index, value) in enumerate(self.directory_lookup_list):
            if idx >= value:
                dataset_index = index
                accumulated = value
            else:
                break

        return (dataset_index, idx - accumulated)

    def get_scene_dirs_sizes(self, scene_dirs: List[str]) -> List[int]:
        """We use the ground truth size as reference for the size of each scene dataset. Also see validate_dataset."""
        sizes = []
        for scene_dir in scene_dirs:
            path = join(scene_dir, BUFFER_TYPE_TO_NAME[BufferType.GT_RTGI], self.dynamic_range, self.resolution)
            num_files_in_path = len(self.all_filepaths_for_files_in_path(path))
            sizes.append(num_files_in_path)

        return sizes

    # [1, 3, 5, 7] -> [0, 1, 4, 9]
    def accumulated_sizes_per_scene(self, num_images_per_scene: List[int]) -> List[int]:
        # itertools.accumulated should have been ideal, but we shift the whole list to the right
        accumulated_sizes = []
        accumulated_size = 0

        for num_images in num_images_per_scene:
            accumulated_sizes.append(accumulated_size)
            accumulated_size += num_images  # ignored on the last iteration

        return accumulated_sizes

    def calculate_dataset_size(self) -> int:
        """We use the rtgi-<dynimic_range>-<resolution> combination as reference for the size of each scene dataset."""
        scene_dirs = self.get_scene_dirs()
        size = 0
        for scene_dir in scene_dirs:
            path = join(scene_dir, BUFFER_TYPE_TO_NAME[BufferType.GT_RTGI], self.dynamic_range, self.resolution)
            files = self.all_filepaths_for_files_in_path(path)
            size += len(files)

        return size

    def validate_dataset(self):
        """We validate against the rtgi ground truth (GT) images. For all GT images there should be an image from
        all input buffers used."""

        print(
            "Dataset to be validated against "
            + " ".join([BUFFER_TYPE_TO_NAME[input_buffer] for input_buffer in self.input_buffers])
        )

        scene_dirs = self.get_scene_dirs()
        is_valid = True
        for scene_dir in scene_dirs:
            gt_path = join(scene_dir, BUFFER_TYPE_TO_NAME[BufferType.GT_RTGI], self.dynamic_range, self.resolution)
            gt_files = self.all_filepaths_for_files_in_path(gt_path)
            validation_size = len(gt_files)
            for input_buffer in self.input_buffers:
                buffer_dir_path = join(
                    scene_dir, BUFFER_TYPE_TO_NAME[input_buffer], self.dynamic_range, self.resolution
                )
                files_in_buffer_dir = self.all_filepaths_for_files_in_path(buffer_dir_path)
                if not set(gt_files).issubset(set(files_in_buffer_dir)):
                    print("[Error] There are GT files not in " + buffer_dir_path + "! Dataset invalid!")
                    is_valid = False
                if len(files_in_buffer_dir) != validation_size:
                    print("[Warning] Dataset " + buffer_dir_path + " length is not matching the RTGI length")

        if is_valid:
            print("Dataset is valid.")
        else:
            raise Exception("Dataset is not valid.")

    def get_scene_dirs(self) -> List[str]:
        return [join(self.root_dir, f) for f in listdir(self.root_dir) if isdir(join(self.root_dir, f))]

    def all_filepaths_for_files_in_path(self, path) -> List[str]:
        return [f for f in listdir(path) if isfile(join(path, f))]
