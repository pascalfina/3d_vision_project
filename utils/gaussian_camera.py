import numpy as np
import torch
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


def build_minicam(
    width: int,
    height: int,
    fovy: float,
    fovx: float,
    znear: float,
    zfar: float,
    R,
    T,
    device: str = "cuda",
) -> MiniCam:
    # Match the older Camera-style R/T inputs while constructing the newer MiniCam API.
    world_view_transform = torch.tensor(
        getWorld2View2(np.asarray(R), np.asarray(T)), dtype=torch.float32, device=device
    ).transpose(0, 1)
    projection_matrix = getProjectionMatrix(
        znear=znear, zfar=zfar, fovX=fovx, fovY=fovy
    ).transpose(0, 1).to(device)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    return MiniCam(
        width=width,
        height=height,
        fovy=fovy,
        fovx=fovx,
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view_transform,
        full_proj_transform=full_proj_transform,
    )
