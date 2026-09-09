import torch
import torch.utils.data
from .unimatch.unimatch import UniMatch
from .unimatch.unimatch_depthsplat import UniMatchDepthSplat
from .unimatch.dataloader.stereo import transforms as stereo_transforms

from .unimatch.dataloader.depth import augmentation as depth_transforms

from .unimatch.utils.utils import InputPadder
import torch.nn.functional as F
import numpy as np
import cv2
import os
code_dir = os.path.dirname(os.path.realpath(__file__))


class UnimatchModel():
    def __init__(self, 
                 device='cuda',
                 task='stereo',
                 **kwargs
                 ):
        self.device = device
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.task = task
        if task == 'stereo':
            self.model = UniMatch(feature_channels=128,
                                    num_scales=2,
                                    upsample_factor=4,
                                    num_head=1,
                                    ffn_dim_expansion=4,
                                    num_transformer_layers=6,
                                    reg_refine=False,
                                    task='stereo').to(device)

            
            ckpt_path = os.path.join(code_dir, '../model_data/ckpt/unimatch/gmstereo-scale2-resumeflowthings-sceneflow-48020649.pth')
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint['model'])
            self.model.eval()

        elif task == 'flow':
            self.model = UniMatch(feature_channels=128,
                                    num_scales=2,
                                    upsample_factor=8,
                                    num_head=1,
                                    ffn_dim_expansion=4,
                                    num_transformer_layers=6,
                                    reg_refine=False,
                                    task='flow').to(device)

            ckpt_path = os.path.join(code_dir, '../model_data/ckpt/unimatch/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth')
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.model.eval()

        elif task == 'depth':
            # self.model = UniMatchDepthSplat(num_scales=1,
            #                                 upsample_factor=8,
            #                                 vit_type='vits',
            #                                 ).to(device)
            
            # ckpt_path = 'core/unimatch/ckpt/depthsplat-depth-small-352x640-randview2-8-e807bd82.pth'
            self.model = UniMatch(feature_channels=128,
                                num_scales=1,
                                upsample_factor=8,
                                num_head=1,
                                ffn_dim_expansion=4,
                                num_transformer_layers=6,
                                reg_refine=True,
                                task='depth').to(device)


            ckpt_path = os.path.join(code_dir, '../model_data/ckpt/unimatch/gmdepth-scale1-regrefine1-resumeflowthings-scannet-90325722.pth')
            checkpoint = torch.load(ckpt_path)
            self.model.load_state_dict(checkpoint['model'])
            self.model.eval()


        self.stereo_transform = stereo_transforms.Compose([stereo_transforms.ToTensor(),
                                                stereo_transforms.Normalize(mean=self.IMAGENET_MEAN, 
                                                                     std=self.IMAGENET_STD)
                                                ])
        

        self.depth_transform = depth_transforms.Compose([depth_transforms.ToTensor(),
                                                depth_transforms.Normalize(mean=self.IMAGENET_MEAN, 
                                                                     std=self.IMAGENET_STD)
                                                ])
        

    def get_disparity(self, imgL, imgR):
        if self.task == 'stereo':
            left = imgL.astype(np.float32)
            right = imgR.astype(np.float32)
            sample = {'left': left, 'right': right}

            sample = self.stereo_transform(sample)

            left = sample['left'].to(self.device).unsqueeze(0)  # [1, 3, H, W]
            right = sample['right'].to(self.device).unsqueeze(0)  # [1, 3, H, W]

            padder = InputPadder(left.shape, padding_factor=32)
            left, right = padder.pad(left, right)
            with torch.no_grad():
                pred_disp = self.model(left, right,
                                    attn_type='self_swin2d_cross_swin1d',
                                    attn_splits_list=[2, 8],
                                    corr_radius_list=[-1, 4], ###
                                    prop_radius_list=[-1, 1],
                                    num_reg_refine=1,
                                    task='stereo',
                                    )  # [1, H, W]
                


            pred_disp = padder.unpad(pred_disp)[0]  # [H, W]

            # print(pred_disp.shape)
            disparity = pred_disp.detach().cpu().numpy()
            return disparity
        elif self.task == 'flow':
            flow_uv = self.get_flow(imgL, imgR)
            u = flow_uv[:,:,0]
            # v = flow_uv[:,:,1]
            # disparity = np.sqrt(np.square(u) + np.square(v))

            return np.abs(u)
    

    def get_flow(self, imgL, imgR):
        # print(imgL.shape)
        ori_size = imgL.shape[:2]
        imgL = cv2.resize(imgL, (int(imgL.shape[1] * 0.5), int(imgL.shape[0] * 15 / 31)))
        imgR = cv2.resize(imgR, (int(imgR.shape[1] * 0.5), int(imgR.shape[0] * 15 / 31)))
        img1 = torch.from_numpy(imgL[..., :3]).permute(2, 0, 1).float()
        img2 = torch.from_numpy(imgR[..., :3]).permute(2, 0, 1).float()
        # print(img1.shape)
        image1 = img1[None].to(self.device)
        image2 = img2[None].to(self.device)
        # ori_size = image1.shape[-2:]
        padder = InputPadder(image1.shape, padding_factor=32)
        image1, image2 = padder.pad(image1, image2)
        # print(img1.shape)
        results_dict = self.model(image1, image2,
                                 attn_type='swin',
                                 attn_splits_list=[2, 8],
                                 corr_radius_list=[-1, 4],
                                 prop_radius_list=[-1, 1],
                                 num_reg_refine=6,
                                 task='flow',
                                 pred_bidir_flow=True,
                                 )
        
        flow_pr = results_dict
        
        
        flow_pr = padder.unpad(flow_pr[0])
        flow_pr = F.interpolate(flow_pr.unsqueeze(0), size=ori_size, mode='bilinear',
                                        align_corners=True)
        flow = flow_pr[0].detach().permute(1, 2, 0).cpu().numpy()
        # print(flow.shape)
        flow_pr[:, 0] = flow_pr[:, 0] * ori_size[0] / flow.shape[0]
        flow_pr[:, 1] = flow_pr[:, 1] * ori_size[1] / flow.shape[1]
        # print(ori_size[0] / flow.shape[0], ori_size[1] / flow.shape[1])
        # exit()
        return -flow
    

    def get_threeD(self, imgL, imgR, pose, intrinsics):
        
        # img_ref = imgL.astype('float32')
        # img_tgt = imgR.astype('float32')

        sample = {
            'img_ref': imgL,
            'img_tgt': imgR,
            'intrinsics': intrinsics,
            'pose': pose,
            'depth': None,
            'valid': None,
        }
        K = intrinsics

        img_ref = torch.from_numpy(sample['img_ref']).to(self.device).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
        img_tgt = torch.from_numpy(sample['img_tgt']).to(self.device).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
        intrinsics = torch.from_numpy(sample['intrinsics']).to(self.device).unsqueeze(0).float()  # [1, 3, 3]
        pose = torch.from_numpy(sample['pose']).to(self.device).unsqueeze(0).float()  # [1, 4, 4]



        # print(img_ref.shape, intrinsics.shape, pose.shape)
        padder = InputPadder(img_ref.shape, padding_factor=32, mode='kitti')
        img_ref, img_tgt = padder.pad(img_ref, img_tgt)

        # print(img_ref.shape, intrinsics.shape, pose.shape)

        sample = self.depth_transform(sample)

        pred_depth = self.model(img_ref, img_tgt,
                                attn_type='swin',
                                attn_splits_list=[2],
                                prop_radius_list=[-1],
                                num_reg_refine=1,
                                intrinsics=intrinsics,
                                pose=pose,
                                min_depth=0.1,
                                max_depth=10,
                                num_depth_candidates=64,
                                task='depth',
                                )  # [1, H, W]


        depth = padder.unpad(pred_depth[0]).detach().cpu().numpy()
        # depth = padder.unpad(pred_depth[0]).permute(1, 0).detach().cpu().numpy()
        H, W = depth.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # 生成像素坐标网格 (u, v)
        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)  # (H, W)

        # 计算相机坐标系下的X, Y, Z
        Z = depth
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        threeD = np.stack([X, Y, Z], axis=-1)

        return threeD
    
        # return depth