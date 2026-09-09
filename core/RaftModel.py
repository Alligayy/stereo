import sys
import torch
from .raft_stereo_core.raft_stereo import RAFTStereo
from .raft_stereo_core.utils.utils import InputPadder
import os
from utils.logger import get_logger

logger = get_logger("core.RaftModel")

code_dir = os.path.dirname(os.path.realpath(__file__))

class RAFTParams:
    def __init__(self, **classes):
        for attr_name, cls in classes.items():
            setattr(self, attr_name, cls)


class RAFTStereoModel():
    def __init__(self, 
                 restore_ckpt_folder=os.path.join(code_dir, '../model_data/ckpt/raft'),
                 mode='realtime',
                 mixed_precision=True,
                 corr_implementation = '',
                 valid_iters = 18,
                #  onnx_model_path = '',
                 device = 'cpu',
                 **kwargs
                 ):
        
        if not corr_implementation:
            try:
                import corr_sampler # type: ignore
                logger.info('corr_sampler 导入成功！使用加速采样的 sampler!')
                corr_implementation = 'reg_cuda'
            except:
                corr_implementation = 'reg'

            
                

        self.device = device
        sys.path.append('core')
        self.paramdict = {'mixed_precision':mixed_precision,
                          'hidden_dims':[128]*3,
                          'corr_implementation':corr_implementation,
                          'corr_levels':4,
                          'corr_radius':4,
                          'context_norm':'batch',
                          'slow_fast_gru':4,
                          'valid_iters':valid_iters,}
        if mode == 'realtime':
            restore_ckpt = os.path.join(restore_ckpt_folder, 'raftstereo-realtime.pth')
            self.paramdict['n_downsample'] = 3
            self.paramdict['n_gru_layers'] = 2
            self.paramdict['shared_backbone'] = True
        else:
            if mode == 'eth3d':
                restore_ckpt = os.path.join(restore_ckpt_folder, 'raftstereo-eth3d.pth')
            elif mode == 'middlebury':
                restore_ckpt = os.path.join(restore_ckpt_folder, 'raftstereo-middlebury.pth')
            elif mode == 'flow':
                restore_ckpt = os.path.join(restore_ckpt_folder, 'raftstereo-sceneflow.pth')
            else:
                logger.error('No such mode to initialize the model!')
                sys.exit(1)

            self.paramdict['n_downsample'] = 2
            self.paramdict['n_gru_layers'] = 3
            self.paramdict['shared_backbone'] = False

        params = type('', (), {})
        for attr_name, cls in self.paramdict.items():
            setattr(params, attr_name, cls)
        if self.device == 'cpu':
            self.model = RAFTStereo(params)
            checkpoint = torch.load(restore_ckpt, map_location=torch.device(self.device))

            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith('module.'):
                    k = k[len('module.'):]
                new_state_dict[k] = v

            self.model.load_state_dict(new_state_dict)
        else:
            self.model = torch.nn.DataParallel(RAFTStereo(params), device_ids=[0])
            self.model.load_state_dict(torch.load(restore_ckpt))
        
            self.model = self.model.module
        self.model.to(self.device).eval()

        
        

    def get_disparity(self, image1, image2):
        
        
        with torch.no_grad():

            image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0)
            image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0)
            self.padder = InputPadder(image1.shape, divis_by=8)
            image1, image2 = self.padder.pad(image1, image2)
            
            flow_up = self.model(image1.to(self.device), image2.to(self.device), iters=self.paramdict['valid_iters'], test_mode=True)
            
            flow_up = self.padder.unpad(flow_up).squeeze()
            disparity = flow_up.detach().cpu().numpy()


        return disparity