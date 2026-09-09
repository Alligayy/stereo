import os
import torch
import sys
import numpy as np
import onnx
from onnxsim import simplify
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{code_dir}/../')
from core.model import Model
from utils.logger import get_logger

logger = get_logger("utils.make_onnx")


def export_onnx(model_name, save_path, height, width, onnx_simplify=False):
    logger.info(f'当前导出模型名称：{model_name}')
    os.environ['XFORMERS_DISABLED'] = '1'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    device = 'cuda' if model_name in ['foundation', 'defom'] else 'cpu'
    torch.autograd.set_grad_enabled(False)
    model = Model(model_name, flash=False, device=device).model.model
    # model = model.half()
    if model_name in ['foundation', 'defom']:
        left_img = torch.randn(1, 3, height, width).cuda().float()
        right_img = torch.randn(1, 3, height, width).cuda().float()
    else:
        left_img = torch.randn(1, 3, height, width).float()
        right_img = torch.randn(1, 3, height, width).float()

    torch.onnx.export(
        model,
        (left_img, right_img),
        save_path,
        opset_version=18,
        input_names = ['left', 'right'],
        output_names = ['disp'],
        dynamic_axes={
            'left': {0 : 'batch_size'},
            'right': {0 : 'batch_size'},
            'disp': {0 : 'batch_size'}
        },
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
    )
    if onnx_simplify:
        
        logger.info('正在简化onnx模型')
        model_ox = onnx.load(save_path)
        model_simp, check = simplify(model_ox)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, save_path)
        logger.info('onnx模型简化成功！')

    logger.info(f'{model_name} 模型onnx导出完毕！')
    del model



if __name__ == '__main__':

    # 模型名称 
    # model_name = 'foundation'
    for model_name in ['foundation', 'monster', 'defom', 'cre', 'psm', 'raft', 'sea-raft', ]:
        export_onnx(model_name,  f'./model_data/onnx/{model_name}_stereo/{model_name}_stereo.onnx', 480, 640)
    