from torch.utils import data
from GIDataset import DYNAMIC_RANGE_KEY, GIDataset, INPUT_BUFFERS_KEY, RESOLUTION_KEY, SIZE_KEY

import json
from typing import List
from GIDataset import BufferType
from os.path import join
from os import listdir
import torch


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


class GIDatasetCached:
    def __init__(self, cached_dir: str, input_buffers: List[BufferType], use_hdr: bool, resolution: int) -> None:
        """
        The useHDR and resolution parameters are used just for validation that the cached files are
        of the expected type. input_buffers
        """
        self.cached_dir = cached_dir

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
            raise Exception("Dynamic range in cached set is {} but use_hdr flag is {}".format(cached_dynamic_range, use_hdr))
        
        self.tensor_filepaths = [join(cached_dir, filename) for filename in listdir(cached_dir) if filename.endswith(".pt")]
        actual_size = len(self.tensor_filepaths)
        if actual_size != self.size:
            raise Exception("Dataset sizes don't match, description promises for {} items but there are {} items in directory".format(self.size, actual_size))
        
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.load(self.tensor_filepaths[idx])

