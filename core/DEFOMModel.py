import sys
sys.path.append('core')

import numpy as np
import torch
from defom_core.defom_stereo import DEFOMStereo
from defom_core.utils.utils import InputPadder
import os
from utils.logger import get_logger

logger = get_logger("core.DEFOMModel")

code_dir = os.path.dirname(os.path.realpath(__file__))

class DEFOMStereoModel:

    def __init__(self, 
                #  ckpt=os.path.join(code_dir, 'defom_core/ckpt/defomstereo_vitl_eth3d.pth'),
                #  ckpt=r'core\foundation_core\ckpt\model_best_bp2.pth',
                 device='cuda',
                 mixed_precision = True,
                 **kwargs,
                 ):
        param_dict = {
            'mixed_precision':mixed_precision,
            'valid_iters':12,
            'scale_iters':3,
            'dinov2_encoder':'vits',
            'idepth_scale':0.5,
            'hidden_dims':[128]*3,
            'shared_backbone':False,
            'corr_levels':2,
            'corr_radius':4,
            'scale_list':[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
            'scale_corr_radius':2,
            'n_downsample':2,
            'context_norm':"batch",
            'n_gru_layers':3,
            'corr_implementation':'reg',
        }

        self.args = type('', (), {})
        for attr_name, cls in param_dict.items():
            setattr(self.args, attr_name, cls)
            # print(attr_name, cls)

        ckpt = os.path.join(code_dir, f'../model_data/ckpt/defom/defomstereo_{self.args.dinov2_encoder}_sceneflow.pth')
        logger.info(f'加载 DEFOM 权重: {ckpt}')

        self.model = DEFOMStereo(self.args)
        checkpoint = torch.load(ckpt, map_location='cuda')
        if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.device = device
        self.model.to(self.device)
        self.model.eval()



    def get_disparity(self, imgL, imgR):

        # print(imgL.shape)
        image1 = torch.from_numpy(imgL).permute(2, 0, 1).float()[None].to(self.device)
        image2 = torch.from_numpy(imgR).permute(2, 0, 1).float()[None].to(self.device)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        
        # print(image1.shape, torch.max(image1))
        with torch.no_grad():
            disp_pr = self.model(image1, image2, iters=self.args.valid_iters, scale_iters=self.args.scale_iters, test_mode=True)
        disp_pr = padder.unpad(disp_pr).cpu().squeeze().numpy()
        # print(disp_pr.shape, np.max(disp_pr), np.min(disp_pr))
        return disp_pr


if __name__ == '__main__':


    model = DEFOMStereoModel()
    

