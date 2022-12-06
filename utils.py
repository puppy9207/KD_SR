import math
import pickle
import numpy as np
import torch

from torch import distributed as dist
from torch.utils.data.sampler import Sampler

def check_image_file(filename: str):
    return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP"])
    
def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()


def synchronize():
    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()

    if world_size == 1:
        return

    dist.barrier()


def get_world_size():
    if not dist.is_available():
        return 1

    if not dist.is_initialized():
        return 1

    return dist.get_world_size()


def reduce_sum(tensor):
    if not dist.is_available():
        return tensor

    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor


def gather_grad(params):
    world_size = get_world_size()

    if world_size == 1:
        return

    for param in params:
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data.div_(world_size)


def all_gather(data):
    world_size = get_world_size()

    if world_size == 1:
        return [data]

    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    local_size = torch.IntTensor([tensor.numel()]).to("cuda")
    size_list = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))

    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), 0)

    dist.all_gather(tensor_list, tensor)

    data_list = []

    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_loss_dict(loss_dict):
    world_size = get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        keys = []
        losses = []

        for k in sorted(loss_dict.keys()):
            keys.append(k)
            losses.append(loss_dict[k])

        losses = torch.stack(losses, 0)
        dist.reduce(losses, dst=0)

        if dist.get_rank() == 0:
            losses /= world_size

        reduced_losses = {k: v for k, v in zip(keys, losses)}

    return reduced_losses

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

# 전처리 과정 함수
def preprocess(img):
    # uInt8 -> float32로 변환
    x = np.array(img).astype(np.float32)
    x = x.transpose([2,0,1])
    # Normalize x 값
    x /= 255.
    # 넘파이 x를 텐서로 변환
    x = torch.from_numpy(x)
    # x의 차원의 수 증가
    x = x.unsqueeze(0)
    # x 값 반환
    return x

# 후처리 과정 함수
def postprocess(tensor):
    x = tensor.mul(255.0).cpu().numpy().squeeze(0)
    x = np.array(x).transpose([1,2,0])
    x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    return x

def get_concat_h(img1, img2):
    merge = pil_image.new("RGB", (img1.width + img2.width, img1.height))
    merge.paste(img1, (0, 0))
    merge.paste(img2, (img2.width, 0))
    return merge

def rgb2yuv(rgb):
    m = np.array([
        [0.29900, -0.147108,  0.614777],
        [0.58700, -0.288804, -0.514799],
        [0.11400,  0.435912, -0.099978]
    ])
    yuv = np.dot(rgb, m)
    yuv[:,:,1:] += 0.5
    return yuv

def yuv2rgb(yuv, gpu):
    m = torch.tensor(
        [
            [1.000,  1.000, 1.000],
            [0.000, -0.394, 2.032],
            [1.140, -0.581, 0.000],
        ]
    ).to(gpu)

    yuv[:, 1:, :, :] -= 0.5
    temp = yuv.permute((0,2,3,1))
    rgb = torch.matmul(temp, m)
    rgb = rgb.permute((0,3,1,2))
    return rgb
