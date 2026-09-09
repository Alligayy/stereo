import torch
from .psm_stereo_core import PSMNet
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
code_dir = os.path.dirname(os.path.realpath(__file__))

class PSMStereoModel():
    def __init__(self, 
                 restore_ckpt=os.path.join(code_dir, '../model_data/ckpt/psm/pretrained_model_KITTI2015.tar'),
                 maxdisp = 256,
                 device = 'cuda',
                 **kwargs,
                 ):
        self.device = device
        torch.cuda.manual_seed(1)

        if self.device == 'cpu':
            self.model = PSMNet(maxdisp=maxdisp, device='cpu')
            checkpoint = torch.load(restore_ckpt, map_location=torch.device(self.device))

            new_state_dict = {}
            for k, v in checkpoint["state_dict"].items():
                if k.startswith('module.'):
                    k = k[len('module.'):]
                new_state_dict[k] = v

            self.model.load_state_dict(new_state_dict)
        else:
            self.model = nn.DataParallel(PSMNet(maxdisp=maxdisp, device='cuda'), device_ids=[0])
            self.model.load_state_dict(torch.load(restore_ckpt)["state_dict"])

        self.model.to(self.device)
        self.model.eval()

        normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                           'std': [0.229, 0.224, 0.225]}
        self.infer_transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(**normal_mean_var)])    

        

    def get_disparity(self, imgL, imgR):

        imgL = torch.from_numpy(imgL).float().permute(2,0,1)  # numpy转tensor
        imgR = torch.from_numpy(imgR).float().permute(2,0,1)
        
        

        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1] // 16       
            top_pad = (times+1) * 16 - imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2] // 16                       
            right_pad = (times+1) * 16 - imgL.shape[2]
        else:
            right_pad = 0    

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0).to(self.device)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0).to(self.device)
        # imgL = self.infer_transform(imgL)
        # imgR = self.infer_transform(imgR) 

        

        # print(imgL.shape, imgR.shape)
        with torch.no_grad():
            disp = self.model(imgL, imgR)
            # print(disp.shape)

        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()
        
        if top_pad != 0 and right_pad != 0:
            disparity = pred_disp[top_pad:, :-right_pad]
        elif top_pad == 0 and right_pad != 0:
            disparity = pred_disp[:, :-right_pad]
        elif top_pad != 0 and right_pad == 0:
            disparity = pred_disp[top_pad:, :]
        else:
            disparity = pred_disp

        
        return disparity

