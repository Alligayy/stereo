import numpy as np
import torch
import torch.nn.functional as F
from .sea_raft_core.raft import RAFT
import json
import os
code_dir = os.path.dirname(os.path.realpath(__file__))


class SeaRaftModel():

    def __init__(self, device='cuda', scale='M', **kwargs):

        self.defaults = {
            'cfg':os.path.join(code_dir, f'sea_raft_core/config/eval/spring-{scale}.json',),
            # 'cfg':f'./core/sea_raft_core/config/eval/sintel-{scale}.json',
            # 'cfg':f'./core/sea_raft_core/config/eval/kitti-{scale}.json',
            # 'model':f'./core/sea_raft_core/ckpt/Tartan-C-T-TSKH432x960-{scale}.pth',
            # 'model':f'core/sea_raft_core/ckpt/Tartan-C-T-TSKH-kitti432x960-{scale}.pth',
            'model':os.path.join(code_dir, f'../model_data/ckpt/sea-raft/Tartan-C-T-TSKH-spring540x960-{scale}.pth',),
            # 'model':f'./core/sea_raft_core/ckpt/Tartan-C-T432x960-{scale}.pth',
            # 'model':f'./core/sea_raft_core/ckpt/Tartan480x640-{scale}.pth',
        }

        with open(self.defaults['cfg'], 'r', encoding='utf-8') as file:
            self.defaults.update(json.load(file))

        self.device = device

        self.args = type('', (), {})
        for attr_name, cls in self.defaults.items():
            setattr(self.args, attr_name, cls)

        self.model = RAFT(self.args)

        state_dict = torch.load(self.args.model, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device)
        self.model.eval()

        # print(self.args.image_size)
        # exit()
# 
    def get_flow(self, image1, image2):
        image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
        image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
        H, W = image1.shape[1:]
        image1 = image1[None].to(self.device)
        image2 = image2[None].to(self.device)

        # image1 = F.interpolate(image1, scale_factor=2 ** self.args.scale, mode='bilinear', align_corners=False)
        # image2 = F.interpolate(image2, scale_factor=2 ** self.args.scale, mode='bilinear', align_corners=False)

        flow = self.model(image1, image2, iters=self.args.iters, test_mode=True)
        # flow = output['flow'][-1]
        # info = output['info'][-1]

        # flow = F.interpolate(flow, scale_factor=0.5 ** self.args.scale, mode='bilinear', align_corners=False) * (0.5 ** self.args.scale)
        # info = F.interpolate(info, scale_factor=0.5 ** self.args.scale, mode='area')
        
        flow = flow[0].permute(1, 2, 0).detach().cpu().numpy()

        return flow

    def get_disparity(self, image1, image2):
        flow_uv = self.get_flow(image1, image2)
        u = flow_uv[:,:,0]
        v = flow_uv[:,:,1]
        disparity = np.sqrt(np.square(u) + np.square(v))
        # return disparity
        return np.abs(u)


def warp_image_flow(img1, flow):
    h, w = img1.shape[:2]
    x1, y1 = np.meshgrid(np.arange(w), np.arange(h))
    
    x2 = (x1 + flow[..., 0]).astype(int)
    y2 = (y1 + flow[..., 1]).astype(int)
    
    mask = (x2 >= 0) & (x2 < w) & (y2 >= 0) & (y2 < h)
    
    warped = np.zeros_like(img1)
    warped[y2[mask], x2[mask]] = img1[y1[mask], x1[mask]]
    
    return warped
