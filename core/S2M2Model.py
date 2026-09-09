import os
import argparse
import numpy as np
import cv2
import torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from .s2m2_core.s2m2 import S2M2
from utils.logger import get_logger
torch.backends.cudnn.benchmark = True

logger = get_logger("core.S2M2Model")

code_dir = os.path.dirname(os.path.realpath(__file__))
class S2M2Model:
    def __init__(self,
                 model_type: str = 'S',
                 num_refine: int = 3,
                 allow_negative: bool = False,
                 torch_compile: bool = False,
                 device: str = 'cuda:0',
                 weights_dir: str = os.path.join(code_dir, '../model_data/ckpt/s2m2'),
                 **kwargs):
        """
        weights_dir: 权重文件所在文件夹，默认为 'pretrain_weights'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.weights_dir = weights_dir
        self.model = self._build_model(model_type, num_refine, allow_negative)
        self.model.to(self.device).eval()
        if torch_compile:
            self.model = torch.compile(self.model)

    # ---------- 内部工具 ----------
    def _build_model(self, model_type, num_refine, allow_negative):
        if model_type == "S":
            feat_ch, n_tr = 128, 1
        elif model_type == "M":
            feat_ch, n_tr = 192, 2
        elif model_type == "L":
            feat_ch, n_tr = 256, 3
        elif model_type == "XL":
            feat_ch, n_tr = 384, 3
        else:
            raise ValueError('model_type must be one of [S, M, L, XL]')

        ckpt_name = f'CH{feat_ch}NTR{n_tr}.pth'
        ckpt_path = os.path.join(self.weights_dir, ckpt_name)
        logger.info(f'Load {ckpt_path}')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'权重文件不存在: {ckpt_path}')

        model = S2M2(feature_channels=feat_ch,
                      dim_expansion=1,
                      num_transformer=n_tr,
                      use_positivity=not allow_negative,
                      refine_iter=num_refine)
        checkpoint = torch.load(ckpt_path, weights_only=True)
        model.my_load_state_dict(checkpoint['state_dict'])
        return model

    @staticmethod
    def _pad_to_multiple32(img: np.ndarray) -> tuple[np.ndarray, int, int]:
        """右下补 0 到 32 倍数，返回 (pad_img, pad_w, pad_h)"""
        h, w = img.shape[:2]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h == 0 and pad_w == 0:
            return img, 0, 0
        pad_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                       cv2.BORDER_CONSTANT, value=0)
        return pad_img, pad_w, pad_h

    @torch.no_grad()
    def get_disparity(self, imgL: np.ndarray, imgR: np.ndarray, conf=0.5) -> np.ndarray:
        """
        输入：opencv 读取的两张 RGB 图，uint8，H×W×3
        输出：视差图，float32，H×W（与原始输入严格对齐）
        """
        orig_h, orig_w = imgL.shape[:2]

        # 1. 补边
        left_pad , pad_w, pad_h = self._pad_to_multiple32(imgL)
        right_pad, _, _         = self._pad_to_multiple32(imgR)

        # 2. numpy -> tensor
        left_ts  = torch.from_numpy(left_pad ).permute(2, 0, 1).unsqueeze(0).half().to(self.device)
        right_ts = torch.from_numpy(right_pad).permute(2, 0, 1).unsqueeze(0).half().to(self.device)
        logger.debug(f'left_ts.shape={left_ts.shape}')
        # 3. 推理
        with torch.amp.autocast(enabled=True, device_type=self.device.type, dtype=torch.bfloat16):
            pred_disp, _, pred_conf = self.model(left_ts, right_ts)
        
        # 4. 裁掉补边
        disp_pad = pred_disp.squeeze().cpu().numpy()          # Hp×Wp
        valid = ((pred_conf.cpu().float() > conf)).squeeze().numpy()
        disp_pad = disp_pad * valid
        if pad_h > 0:
            disp_pad = disp_pad[:-pad_h, :]
        if pad_w > 0:
            disp_pad = disp_pad[:, :-pad_w]
        return disp_pad  # 尺寸 = orig_h × orig_w


# -------------------- demo --------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--left',  default=r'G:\PythonTemplate\RGBD-yolo\testfile\zed_left_1748247842765.png', help='左图路径')
    parser.add_argument('-R', '--right', default=r'G:\PythonTemplate\RGBD-yolo\testfile\zed_right_1748247842765.png', help='右图路径')
    parser.add_argument('--model_type', default='S', choices=['S', 'M', 'L', 'XL'])
    parser.add_argument('--num_refine', default=3, type=int)
    parser.add_argument('--allow_negative', action='store_true')
    parser.add_argument('--torch_compile', action='store_true')
    parser.add_argument('--weights_dir', default='pretrain_weights', help='权重文件夹路径')
    args = parser.parse_args()

    # 1. 实例化
    stereo = S2M2Model(model_type=args.model_type,
                       num_refine=args.num_refine,
                       allow_negative=args.allow_negative,
                       torch_compile=args.torch_compile,
                       weights_dir=args.weights_dir)

    # 2. 读图
    imgL = cv2.cvtColor(cv2.imread(args.left,  cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    imgR = cv2.cvtColor(cv2.imread(args.right, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    imgL = cv2.resize(imgL, (640, 480))
    imgR = cv2.resize(imgR, (640, 480))
    # 3. 推理
    disparity = stereo.get_disparity(imgL, imgR)

    # 4. 可视化
    d_min, d_max = disparity.min(), disparity.max()
    disp_vis = ((disparity - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    cv2.imshow('left-right', np.hstack((imgL[:, :, ::-1], imgR[:, :, ::-1])))
    cv2.imshow(f'disparity  min:{d_min:.2f}  max:{d_max:.2f}', disp_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()