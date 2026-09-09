import torch
import torch.nn.functional as F
import numpy as np
from .cre_stereo_core import CREStereo
import os
code_dir = os.path.dirname(os.path.realpath(__file__))

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]
    

class CREStereoModel():
    def __init__(self,
                 model_path=os.path.join(code_dir, "../model_data/ckpt/cre/crestereo_eth3d.pth"),
                 device='cpu',
                 **kwargs
                 ):
        self.device = device
        self.model = CREStereo(max_disp=256, mixed_precision=False, test_mode=True)
        self.model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)
        self.model.to(self.device).eval()


    def get_disparity(self, left, right, n_iter=10):

        # print("Model Forwarding...")
        imgL = left.transpose(2, 0, 1)
        imgR = right.transpose(2, 0, 1)
        imgL = np.ascontiguousarray(imgL[None, :, :, :])
        imgR = np.ascontiguousarray(imgR[None, :, :, :])

        imgL = torch.tensor(imgL.astype("float32")).to(self.device)
        imgR = torch.tensor(imgR.astype("float32")).to(self.device)
        self.padder = InputPadder(imgL.shape, divis_by=8)
        imgL, imgR = self.padder.pad(imgL, imgR)


        with torch.inference_mode():
            # print(imgL.shape, imgR.shape)
            # print(imgL.shape)
            pred_flow = self.model(imgL, imgR, iters=n_iter, flow_init=None)
            # print(pred_flow.shape)
            pred_flow = self.padder.unpad(pred_flow)

        pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

        return pred_disp