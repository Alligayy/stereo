
import os
code_dir = os.path.dirname(os.path.realpath(__file__))
from core.depth_pro import depth_pro
import torch
from PIL import Image
from core.depth_pro.utils import load_rgb_from_pil
import numpy as np


def depth_to_3d(depth_map, fx, fy=None, cx=None, cy=None):
    if fy is None:
        fy = fx

    if cx is None:
        cx = depth_map.shape[1] // 2
    if cy is None:
        cy = depth_map.shape[0] // 2

    height, width = depth_map.shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    x_coords = (x_coords - cx) / fx
    y_coords = (y_coords - cy) / fy
    z_coords = depth_map

    # 创建一个与深度图尺寸相同的空数组，用于存储 xyz 坐标
    xyz_map = np.zeros((height, width, 3))

    # 将计算出的 x, y, z 坐标存储到 xyz_map 中
    xyz_map[:, :, 0] = x_coords * z_coords
    xyz_map[:, :, 1] = y_coords * z_coords
    xyz_map[:, :, 2] = z_coords

    return xyz_map


class DepthProModel():

    def __init__(self, device='cuda', **kwargs):
        DEFAULT_MONODEPTH_CONFIG_DICT = depth_pro.DepthProConfig(
            patch_encoder_preset="dinov2l16_384",
            image_encoder_preset="dinov2l16_384",
            checkpoint_uri=os.path.join(code_dir, "../model_data/ckpt/depthpro/depth_pro.pt"),
            decoder_features=256,
            use_fov_head=True,
            fov_encoder_preset="dinov2l16_384",
        )
        model, transform = depth_pro.create_model_and_transforms(config=DEFAULT_MONODEPTH_CONFIG_DICT,
                                                                device=device,
                                                                precision=torch.half,)
        model.eval()
        self.model = model
        self.transform = transform
        

    def get_depth(self, image1, f_px=None, **kwargs):
        image = self.transform(Image.fromarray(image1))
        prediction = self.model.infer(image, f_px=f_px) # type: ignore
        depth = prediction["depth"].detach().cpu().numpy()  # Depth in [m].
        if f_px is None:
            focallength_px = float(prediction["focallength_px"].detach().cpu().numpy())
        else:
            focallength_px = f_px
        return depth, focallength_px
    

    def get_threeD(self, image1, f_px=None, f_py=None, cx=None, cy=None, **kwargs):
        depth_map, focallength_px = self.get_depth(image1, f_px)
        f_px = focallength_px if f_px is None else f_px
        f_py = focallength_px if f_py is None else f_py
        threeD = depth_to_3d(depth_map, f_px, f_py, cx, cy)
        return threeD*1000