import sys
sys.path.append('core')
import torch
from monster_core.monster import Monster 
from monster_core.utils.utils import InputPadder
import os
import sys
import numpy as np
code_dir = os.path.dirname(os.path.realpath(__file__))

class NormalizeTensor(object):
    """Normalize a tensor by given mean and std."""
    
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            
        Returns:
            Tensor: Normalized Tensor image.
        """
        # Ensure mean and std have the same number of channels as the input tensor
        Device = tensor.device
        self.mean = self.mean.to(Device)
        self.std = self.std.to(Device)

        # Normalize the tensor
        if self.mean.ndimension() == 1:
            self.mean = self.mean[:, None, None]
        if self.std.ndimension() == 1:
            self.std = self.std[:, None, None]

        return (tensor - self.mean) / self.std


class MonsterStereoModel:
    def __init__(self, 
                restore_ckpt=os.path.join(code_dir, '../model_data/ckpt/monster/mix_all.pth'),
                valid_iters=16,
                dedvice='cpu',
                **kwargs,
                ):
        self.device = dedvice

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'     

        args_dict = {
                        'save_numpy':False, 
                        'mixed_precision':False, 
                        'valid_iters':valid_iters, 
                        'encoder':'vitl', 
                        'hidden_dims':[128, 128, 128], 
                        'corr_implementation':'reg', 
                        'shared_backbone':False, 
                        'corr_levels':2, 
                        'corr_radius':4, 
                        'n_downsample':2, 
                        'slow_fast_gru':False, 
                        'n_gru_layers':3, 
                        'max_disp':192
                    }

        # args = GetParams(**args_dict)

        args = type('', (), {})
        for attr_name, cls in args_dict.items():
            setattr(args, attr_name, cls)

        model = torch.nn.DataParallel(Monster(args), device_ids=[0])

        # assert os.path.exists(args.restore_ckpt)
        checkpoint = torch.load(restore_ckpt)
        ckpt = dict()
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        for key in checkpoint:
            ckpt['module.' + key] = checkpoint[key]

        model.load_state_dict(checkpoint, strict=True)

        model = model.module
        model.to(self.device)
        model.eval()

        self.model = model
        self.args = args
        # output_directory = Path(args.output_directory)
        # output_directory.mkdir(exist_ok=True)

        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # normalize = NormalizeTensor(mean, std)

    
    
    def get_disparity(self, imgL, imgR):
        image1 = torch.from_numpy(imgL).permute(2, 0, 1).float()[None].to(self.device)
        image2 = torch.from_numpy(imgR).permute(2, 0, 1).float()[None].to(self.device)
        padder = InputPadder(image1.shape, divis_by=32)
        image1, image2 = padder.pad(image1, image2)
        disp = self.model(image1, image2, iters=self.args.valid_iters, test_mode=True)
        disp = padder.unpad(disp)
        # file_stem = os.path.join(output_directory, imfile1.split('/')[-1]).replace('.png', '')
        disp = disp.detach().cpu().numpy().squeeze()
        # disp_np = (2.0*disp).astype(np.uint8)
        return disp
    

if __name__ == '__main__':

    import cv2, time
    model = MonsterStereoModel()
    img1 = cv2.imread('imgs/l3.jpg')
    img2 = cv2.imread('imgs/r3.jpg')

    H, W = (0.8*np.array(img1.shape[:2])).astype('int')
    img1 = cv2.resize(img1, (W, H))
    img2 = cv2.resize(img2, (W, H))

    t1 = time.time()
    disp = model.get_disparity(img1, img2)
    disp = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    T = 1/(time.time() - t1)
    cv2.imshow('disp', disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(T)

