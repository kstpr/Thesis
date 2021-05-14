
import torch
from torch.tensor import Tensor
import torchvision.utils as vutils

from timeit import default_timer as timer
import matplotlib.cm as mpcm

# Test random useful junk 

# t = torch.tensor(
#     [
#         [[10, 11], [12, 13]],
#         [[11, 11], [11, 11]],
#         [[12, 15], [16, 18]],
#         [[20, 20], [20, 20]],
#         [[21, 21], [21, 21]],
#         [[30, 30], [30, 30]],
#         [[31, 31], [31, 31]],
#         [[32, 32], [32, 32]],
#         [[40, 40], [40, 40]],
#     ],
#     dtype=torch.float32
# )

device = device = torch.device("cuda:0")
t = torch.randn(32, 3, 512, 512, dtype=torch.float32).to(device)
# t = torch.full((32, 3, 512, 512), 0.0)

print(t.shape)
# mask_lst = [1,1,0,0,0,0,0,1,1]#[1,1,1,0,0,1,1,1,0]
# mask = torch.tensor(mask_lst).long() # 1s and 3s
# s = torch.squeeze(t[mask.nonzero(), :], 1)
# print(s.shape)


def scale_for(t: torch.Tensor):
    begin = timer()
    for i in range(t.size()[0]):
        t[i] -= torch.min(t[i])
        t[i] /= torch.max(t[i])
    print("Scaling with FOR took {}".format(timer() - begin))


def scale_0_1(t: torch.Tensor):
    begin = timer()
    size = t.size()
    t = t.view(t.size(0), -1)
    t -= t.min(1, keepdim=True)[0]
    if not torch.all(t == 0):
        t /= t.max(1, keepdim=True)[0]
    t = t.view(size)
    print("Scaling with tensor ops took {}".format(timer() - begin))


# s = torch.tensor([[((i + j) / 512) for j in range(256)] for i in range(256)])
# print(s.shape)


def as_color_mapped_image(t: Tensor) -> Tensor:
    cm_hot = mpcm.get_cmap("hot")
    t_np = t.numpy()
    t_np = cm_hot(t_np)
    ss = torch.from_numpy(t_np).permute((2, 0, 1))[0:3, :]
    print(ss.shape)
    grid_tensor = vutils.make_grid([ss], nrow=1)
    vutils.save_image(grid_tensor, "asd.png")


# scale_for(t)
# scale_0_1(t)



