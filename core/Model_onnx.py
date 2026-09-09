import onnxruntime as ort
import numpy as np
import os
import cv2
import platform
from utils.logger import get_logger

logger = get_logger("core.Model_onnx")


class Model_ONNX:
    def __init__(self, onnx_path:str, trt_flag=True, fp16=True, **kwargs):
        os.environ['OMP_NUM_THREADS'] = '8' 
        if not os.path.exists(onnx_path) or not onnx_path.endswith('onnx'):
            logger.error(f"onnx_path 无效: {onnx_path}")
            raise ValueError("onnx_path无效！") 


        fsl = onnx_path.split('_')[-1].split('.')[0].split('x')
        try:
            self.height, self.width = [int(fsli) for fsli in fsl][:2]
            
        except:
            self.height = 480
            self.width = 640
        
        
        self.session_options = ort.SessionOptions()
        providers = ort.get_available_providers()
        logger.info(f'检测到的 providers：{providers}')
        if 'TensorrtExecutionProvider' in providers and trt_flag:
            logger.warning('第一次构建 tensorrt 引擎可能需要很长时间...几分钟～几个小时')
            # onnx_file_path = onnx_path[:-5]
            onnx_file_path = os.path.dirname(onnx_path)
            current_os = platform.system()
            logger.info(f'当前操作系统：{current_os}')
            onnx_file_path = os.path.join(onnx_file_path, 'trt-'+current_os)

            logger.info(f'引擎文件保存路径：{onnx_file_path}')
            trt_options = {
                'trt_fp16_enable': fp16,
                'trt_max_workspace_size': 6<<30,  # 6GB工作空间
                'trt_engine_cache_enable': True,    # 启用引擎缓存
                'trt_engine_cache_path': onnx_file_path,
                'trt_builder_optimization_level': 1,
                # 'trt_tactic_sources': '+CUBLAS,+CUDNN',
            }

            self.providers = [
                ('TensorrtExecutionProvider', trt_options),
                # 'CUDAExecutionProvider',
                # 'CPUExecutionProvider'
            ]
        elif 'CUDAExecutionProvider' in providers:
            if trt_flag:
                logger.warning('未找到 TensorrtExecutionProvider，请检查 tensorrt 安装和 tensorrt 版本是否和 onnxruntime 版本对应！')
            logger.info('使用 CUDAExecutionProvider')
            self.providers = ['CUDAExecutionProvider']
        else:
             logger.warning('cuda 和 tensorrt 均不可用！将使用 CPUExecutionProvider')
             self.providers = ['CPUExecutionProvider']                    
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # self.session_options.intra_op_num_threads = 8
        self.session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # try:
            # 尝试加载

        self.session = ort.InferenceSession(
            onnx_path, 
            sess_options=self.session_options,
            providers=self.providers
        )
        

        if 'TensorrtExecutionProvider' not in self.session.get_providers():
            logger.warning("TensorRT 执行提供程序未启用，将使用 CUDA 或 CPU")
        
        # 打印当前使用的提供程序
        logger.info(f"当前使用的执行提供程序: {self.session.get_providers()}")
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.session_options.intra_op_num_threads = 1  # 根据硬件调整线程数
        self.session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    def get_disparity(self, imgL, imgR):
        # print(imgL.shape)
        left_img = cv2.resize(imgL, (self.width, self.height))  # 或者使用上面的resize_image_opencv函数
        right_img = cv2.resize(imgR, (self.width, self.height))
        left_img = left_img.transpose(2, 0, 1).astype(np.float32)# / 255
        right_img = right_img.transpose(2, 0, 1).astype(np.float32)# / 255
        left_img = np.expand_dims(left_img, axis=0)
        right_img = np.expand_dims(right_img, axis=0)
        inputs = {
            'left': left_img,
            'right': right_img
        }

        outputs = self.session.run(['disp'], inputs)
        disp = outputs[0][0][0].astype(np.float32)
        # print(disp.shape)
        disp = np.abs(disp)
        # print(disp.shape)
        disp = cv2.resize(disp, (imgL.shape[1], imgL.shape[0]), interpolation=cv2.INTER_NEAREST) * (imgL.shape[1] / 640)
        return disp
    

    def __del__(self):

        del self.session
        logger.info("ONNX Runtime Session 资源已释放")