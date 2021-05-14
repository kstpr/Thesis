import enum
from torch.utils import data
from torch import Tensor
from torch.utils.data.dataset import Dataset
import numpy as np
from typing import Dict

from GIDataset import (
    ALL_INPUT_BUFFERS_CANONICAL,
    DYNAMIC_RANGE_KEY,
    INPUT_BUFFERS_KEY,
    RESOLUTION_KEY,
    SIZE_KEY,
)

import json
from typing import List, Tuple
from GIDataset import BufferType
from os.path import join
from os import listdir
import torch
import itertools

STR_TO_BUFFER_TYPE = {
    "BufferType.ALBEDO": BufferType.ALBEDO,
    "BufferType.DI": BufferType.DI,
    "BufferType.WS_NORMALS": BufferType.WS_NORMALS,
    "BufferType.CS_NORMALS": BufferType.CS_NORMALS,
    "BufferType.CS_POSITIONS": BufferType.CS_POSITIONS,
    "BufferType.DEPTH": BufferType.DEPTH,
    "BufferType.GT_RTGI": BufferType.GT_RTGI,
}


def as_buffer_type_enum(d):
    if "__enum__" in d:
        name, member = d["__enum__"].split(".")
        return getattr(BufferType, member)
    else:
        return d


class DatasetType(enum.Enum):
    PYTORCH_TENSORS = 1
    ND_ARRAYS = 2


extensions: Dict[DatasetType, str] = {DatasetType.PYTORCH_TENSORS: "pt", DatasetType.ND_ARRAYS: "npy"}


class GIDatasetCached(Dataset):
    def __init__(
        self,
        cached_dir: str,
        input_buffers: List[BufferType],
        use_hdr: bool,
        resolution: int,
        type: DatasetType = DatasetType.PYTORCH_TENSORS,
    ) -> None:
        """
        The useHDR and resolution parameters are used just for validation that the cached files are
        of the expected type. input_buffers is the list of all buffers besides GT that should be
        returned as a concatenated result, in that order.
        """
        self.cached_dir = cached_dir
        self.buffers_mask = self.mask_as_tensor(self.get_mask(input_buffers))
        self.len_input_buffers = self.input_buffers_len(input_buffers)
        self.dataset_type = type

        with open(join(cached_dir, "dataset_descr.json"), "r") as fp:
            dataset_description = json.load(fp, object_hook=as_buffer_type_enum)

        ### Validation ###
        self.size = int(dataset_description[SIZE_KEY])

        cached_buffers = dataset_description[INPUT_BUFFERS_KEY]
        if BufferType.GT_RTGI in input_buffers:
            raise Exception("GT buffer should not be input.")

        for input_buffer in input_buffers:
            if not input_buffer in cached_buffers:
                raise Exception("Input buffer {} not cached. Cached buffers: {}".format(input_buffer, cached_buffers))

        if int(dataset_description[RESOLUTION_KEY]) != resolution:
            raise Exception(
                "Expected resolution {} but cached files resolution is {}".format(
                    dataset_description[RESOLUTION_KEY], resolution
                )
            )

        cached_dynamic_range = dataset_description[DYNAMIC_RANGE_KEY]
        hdr_check = use_hdr and (cached_dynamic_range == "hdr")
        ldr_check = (not use_hdr) and (cached_dynamic_range == "ldr")
        if not (hdr_check or ldr_check):
            raise Exception(
                "Dynamic range in cached set is {} but use_hdr flag is {}".format(cached_dynamic_range, use_hdr)
            )

        extension = extensions[self.dataset_type]
        self.filepaths = [join(cached_dir, filename) for filename in listdir(cached_dir) if filename.endswith(extension)]
        actual_size = len(self.filepaths)
        if actual_size != self.size:
            raise Exception(
                "Dataset sizes don't match, description promises for {} items but there are {} items in directory".format(
                    self.size, actual_size
                )
            )

    def get_mask(self, input_buffers: List[BufferType]) -> List[int]:
        """Returns a boolean mask for the buffers from the canonical list that should be present in
        the return tensor. No permutation to match the input list order."""
        masked_indices_unflattened = [
            [1] * self.len_buffer(buffer_type) if buffer_type in input_buffers else [0] * self.len_buffer(buffer_type)
            for buffer_type in ALL_INPUT_BUFFERS_CANONICAL
        ] + [
            [1, 1, 1]
        ]  # GT
        flattened = list(itertools.chain.from_iterable(masked_indices_unflattened))
        return flattened

    def len_buffer(self, buffer: BufferType) -> int:
        return 1 if buffer == BufferType.DEPTH else 3

    def mask_as_tensor(self, mask_lst: List[int]) -> Tensor:
        return torch.tensor(mask_lst).long()

    def input_buffers_len(self, input_buffers: List[BufferType]) -> int:
        return sum([self.len_buffer(buffer) for buffer in input_buffers])

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dataset_type == DatasetType.PYTORCH_TENSORS:
            cached_tensor = torch.load(self.filepaths[idx])
        elif self.dataset_type == DatasetType.ND_ARRAYS:
            with open(self.filepaths[idx], 'rb') as f:
                cached_array = np.load(f)
                cached_tensor = torch.from_numpy(cached_array)

        t = cached_tensor[self.buffers_mask.nonzero(), :]
        s = torch.squeeze(t, 1)
        return (s[0 : self.len_input_buffers, :], s[self.len_input_buffers : self.len_input_buffers + 3, :])
