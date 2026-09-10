import os
from omegaconf import OmegaConf
from .foundation_core.utils.utils import InputPadder
from .foundation_core.utils.utils2 import *
from .foundation_core.foundation_stereo import *
from .RaftModel import RAFTParams
from utils.logger import get_logger

logger = get_logger("core.FoundationModel")

code_dir = os.path.dirname(os.path.realpath(__file__))
class FoundationStereoModel:
    def __init__(self, 
                restore_ckpt=os.path.join(code_dir, '../model_data/ckpt/foundation/23-51-11/model_best_bp2.pth'),
                # restore_ckpt=os.path.join(code_dir, '../model_data/ckpt/foundation/11-33-40/model_best_bp2.pth'),
                valid_iters=64,
                # dedvice='cpu',
                **kwargs,
                ):

        logger.info(f'加载 FoundationStereo 权重: {restore_ckpt}')
        args = {
            'scale':1,
            'z_far':10,
            'valid_iters':valid_iters,
            'get_pc':1,
            'hiera':0,
            'small_ratio':1,
        }

        args = RAFTParams(**args)
        # set_logging_format()
        set_seed(0)
        torch.autograd.set_grad_enabled(False)
        # os.makedirs(args.out_dir, exist_ok=True)
        cfg = OmegaConf.load(f'{os.path.dirname(restore_ckpt)}/cfg.yaml')
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        for k in args.__dict__:
            cfg[k] = args.__dict__[k]
        args = OmegaConf.create(cfg)
        # logger.info(f"args:\n{args}")
        logger.info(f"Using pretrained model from {restore_ckpt}")
         
        model = FoundationStereo(args, **kwargs)

        ckpt = torch.load(restore_ckpt, weights_only=False)
        # logger.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
        model.load_state_dict(ckpt['model'])

        model.cuda()
        model.eval()
        # model = torch.jit.load('foundation.pt')

        self.model = model
        self.args = args
                

    def get_disparity(self, imgL, imgR):
        img0 = torch.as_tensor(imgL).cuda().half()[None].permute(0,3,1,2)
        img1 = torch.as_tensor(imgR).cuda().half()[None].permute(0,3,1,2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0, img1 = padder.pad(img0, img1)
        # logger.debug(f'img0.shape={img0.shape}')
        with torch.amp.autocast('cuda', enabled=True):
            if not self.args.hiera:
                disp = self.model.forward(img0, img1, iters=self.args.valid_iters, test_mode=True)
            else:
                disp = self.model.run_hierachical(img0, img1, iters=self.args.valid_iters, test_mode=True, small_ratio=self.args.small_ratio)
        disp = padder.unpad(disp.float())
        disp = disp.data.cpu().numpy()[0, 0]
        # print(disp.shape)
        # exit()
        return disp