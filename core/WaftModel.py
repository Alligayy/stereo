import os
import numpy as np
import torch
from .waft_core.model.vitwarp_v8 import ViTWarpV8
import os
import json
code_dir = os.path.dirname(os.path.realpath(__file__))

class WAFTModel():

    def __init__(self, device='cuda', **kwargs):
        self.defaults = {
            'cfg':os.path.join(code_dir, f'waft_core/config/tar-c-t-spring-540p.json',),
            'ckpt':os.path.join(code_dir, f'../model_data/ckpt/waft/tar-c-t-spring-540p.pth',),
            # 'dataset': 'spring',
        }

        with open(self.defaults['cfg'], 'r', encoding='utf-8') as file:
            self.defaults.update(json.load(file))

        self.device = device
        self.args = type('', (), {})
        for attr_name, cls in self.defaults.items():
            setattr(self.args, attr_name, cls)

        model = ViTWarpV8(self.args)
        state_dict = torch.load(self.args.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        self.model = model

    def get_flow(self, image1, image2):
        image1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
        image2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
        image1 = image1[None].to(self.device)
        image2 = image2[None].to(self.device)
        flow = self.model.forward(image1, image2)
        # flow = output['flow'][-1]
        flow = flow[0].permute(1, 2, 0).detach().cpu().numpy()

        return flow
    
    def get_disparity(self, image1, image2):
        flow_uv = self.get_flow(image1, image2)
        u = flow_uv[:,:,0]
        return np.abs(u)


if __name__ == '__main__':
    model  = WAFTModel()
