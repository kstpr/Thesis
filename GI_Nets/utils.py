import numpy as np
import torch

def tosRGB(image_arr):
    image_arr = np.clip(image_arr, 0, 1)
    # https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
    # from https://gist.github.com/jadarve/de3815874d062f72eaf230a7df41771b
    return np.where(image_arr <= 0.0031308, 12.92 * image_arr, 1.055 * np.power(image_arr, 1 / 2.4) - 0.055)


def scale_0_1_(t: torch.Tensor):
    size = t.size()
    t = t.view(t.size(0), -1)
    t -= t.min(1, keepdim=True)[0]
    if not torch.all(t == 0.0):
        t /= t.max(1, keepdim=True)[0] + 0.0000001
    t = t.view(size)