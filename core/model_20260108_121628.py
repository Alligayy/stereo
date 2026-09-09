import cv2
import numpy as np
import time
import torch
import os
from datetime import datetime
import importlib
from utils.CameraConfig import CameraConfig

code_dir = os.path.abspath(__file__)
code_dir_rl = os.path.dirname(os.path.relpath(code_dir, os.getcwd())) 
code_dir_rl_lib = code_dir_rl.replace('/', '.').replace('\\', '.')

MODEL_CLASSES = \
{
    'cre': code_dir_rl_lib + '.CREModel.CREStereoModel',
    'raft': code_dir_rl_lib + '.RaftModel.RAFTStereoModel',
    'sgbm': code_dir_rl_lib + '.SGBMModel.SGBMStereoModel',
    'unimatch': code_dir_rl_lib + '.UnimatchModel.UnimatchModel',
    'psm': code_dir_rl_lib + '.PSMModel.PSMStereoModel',
    'sea-raft': code_dir_rl_lib + '.SeaRaftModel.SeaRaftModel',
    'foundation': code_dir_rl_lib + '.FoundationModel.FoundationStereoModel',
    'monster': code_dir_rl_lib + '.MonsterModel.MonsterStereoModel',
    'defom': code_dir_rl_lib + '.DEFOMModel.DEFOMStereoModel',
    'depthpro': code_dir_rl_lib + '.DepthProModel.DepthProModel',
    'adcensus': code_dir_rl_lib + '.ADCensusModel.ADCensusModel',
    'waft': code_dir_rl_lib + '.WaftModel.WAFTModel',
    's2m2': code_dir_rl_lib + '.S2M2Model.S2M2Model',
}

ONNX_PATHS = \
{
    'cre':'model_data/onnx/cre_stereo',
    'foundation':'model_data/onnx/foundation_stereo',
    'psm':'model_data/onnx/psm_stereo',
    'raft':'model_data/onnx/raft_stereo',
    'sea-raft':'model_data/onnx/sea-raft_stereo',
    'monster':'model_data/onnx/monster_stereo',
    'defom': 'model_data/onnx/defom_stereo',
    'waft': 'model_data/onnx/waft',
    's2m2': 'model_data/onnx/s2m2',
}

def get_model_class(model_name):
    model_path = MODEL_CLASSES.get(model_name)
    if model_path is None:
        print(f'ERROR:No such mode to initialize the model!\nSupport models: {list(MODEL_CLASSES.keys())}')
        exit(-1)
    module_path, class_name = model_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    return model_class


class Model():

    def __init__(self, 
                 model_name='cre', 
                 device='cuda', 
                 camera_config=None, 
                 onnx=False,
                 onnx_shape=(480, 640),
                 **params):
        if model_name in ['sgbm', 'adcensus']:
            onnx = False
        self.device = device if device in ['cpu', 'cuda'] else 'cuda' if torch.cuda.is_available() else 'cpu'
        params['device'] = self.device
        self.model_name = model_name
        self.params = params
        # onnx 模式输入，优先检查 onnx 路径和 tensorrt 推理模式
        # onnx不存在就导出
        self.onnx = onnx
        if onnx:
            onnx_path = ONNX_PATHS[model_name]
            export_flag = False
            if not os.path.exists(onnx_path):
                export_flag = True
                os.makedirs(onnx_path, exist_ok=True)
            else:
                files = os.listdir(onnx_path)
                sizes = []
                for file in files:
                    fsl = file.split('_')[-1].split('.')[0].split('x')
                    try:
                        get_size = tuple([int(fsli) for fsli in fsl])
                    except:
                        get_size = ()
                    sizes.append(get_size)
                if not onnx_shape in sizes:
                    export_flag = True
                else:
                    onnx_file = os.path.join(onnx_path, files[sizes.index(onnx_shape)])
                    
            if export_flag:
                print('onnx 模型不存在！尝试导出 onnx 模型')
                h, w = onnx_shape
                onnx_file = os.path.join(onnx_path, f'{self.model_name}_stereo_{h}x{w}.onnx')
                self.export_onnx(onnx_shape[0], onnx_shape[1],
                                 save_path=onnx_file)
            
            from .Model_onnx import Model_ONNX
            # fp16 = False if model_name == 'defom' else True
            self.model = Model_ONNX(onnx_file, **params)
        
        # 普通模式，pytorch 初始化模型
        else:
            model_class = get_model_class(model_name)
            self.model = model_class(**params)

        # 初始化相机内参
        self.task = self.model.task if model_name == 'unimatch' else 'stereo'
        self.fps = 0

        if isinstance(camera_config, CameraConfig):
            self.camera_config = camera_config
        else:
            self.camera_config = CameraConfig(camera_config)


        self.mask_flag = True if self.camera_config.Q is not None else False

        # 获取一个畸变校正的掩码（有效掩码）
        if self.mask_flag:
            mask_l = np.ones((self.camera_config.size[1], self.camera_config.size[0])).astype('uint8')
            mask_r = mask_l.copy()
            mask_l, mask_r = self.camera_config.remap(mask_l, mask_r)
            mask = ((mask_l > 0) & (mask_r > 0)).astype('uint8')
            self.mask_origin = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2).astype('bool')
        else:
            self.mask_origin = None

        self.analysis_time = analysis_time_weights

    def get_disparity(self, image1, image2, rate=1):

        if 0 < rate < 1 and not self.onnx:
            image1_r = cv2.resize(image1, (int(image1.shape[1]*rate), int(image1.shape[0]*rate)))
            image2_r = cv2.resize(image2, (int(image2.shape[1]*rate), int(image2.shape[0]*rate)))
            disparity = self.model.get_disparity(image1_r, image2_r)
            disparity = cv2.resize(disparity, (image2.shape[1], image2.shape[0]), interpolation=cv2.INTER_NEAREST) / rate
        else:
            disparity = self.model.get_disparity(image1, image2)
        
        if not self.mask_flag:
            self.mask = np.ones((image1.shape[0], image1.shape[1])).astype('bool')
            
        else:
            # mask = self.mask_origin.copy()
            disparity = np.where(self.mask_origin, disparity, 0).astype('float32')

        return disparity
    
    def update(self, image1, image2, rate=1, remap=True, colormap=cv2.COLORMAP_INFERNO, max_z=10000, show_text=True):
        

        res = {
                'disparity' : None,
                'threeD' : None,
                'disp_gray' : None,
                'disp_map' : None,
                'image1' : None,
                'image2' : None,
                'image1_org' : None,
                'image2_org' : None,
                'points': None,
                'framecolor': None,
                'mask':None,
                'time':{},
                }

        # 获得视差
        if self.model_name == 'unimatch' and self.task == 'depth' and self.mask_flag: 
            res['image1'], res['image2'] = image1.copy(),image2.copy()
            K = self.camera_config.left_camera_matrix * rate
            pose = self.camera_config.c2w.copy()   
            pose[:3, 3] = - np.abs(pose[:3, 3])
            pose[:3, :3] = pose[:3, :3].T
            threeD = self.model.get_threeD(image1, image2, pose, K)

            # print(K, pose)
            threeD = cv2.resize(threeD, (image1.shape[1], image1.shape[0])) / rate
            # self.threeD = threeD
            disp = cv2.normalize(-np.abs(threeD[:, :, 2]), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disp_gray = disp.copy()
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
            disp_map = cv2.applyColorMap(disp, colormap)
            fps = self.get_fps()
            cv2.putText(disp_map, 'fps=%.1f'%fps, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(disp_map, f'model:{self.model_name}', (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # self.disp_map = disp_map


            res['disp_gray'] = disp_gray
            res['disp_map'] = disp_map
            res['threeD'] = threeD


            points = threeD.reshape(-1, 3)
            framecolor = (cv2.cvtColor(image1, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0)

        if self.model_name == 'depthpro':
            res['image1'] = image1
            res['image2'] = np.zeros_like(image1)
            if self.mask_flag:
                fx = self.camera_config.left_camera_matrix[0, 0]
                fy = self.camera_config.left_camera_matrix[1, 1]
            else:
                fx = None
                fy = None

            threeD = self.model.get_threeD(image1, fx, fy)
            disp = cv2.normalize(-np.abs(threeD[:, :, 2]), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disp_gray = disp.copy()
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
            disp_map = cv2.applyColorMap(disp, colormap)
            fps = self.get_fps()
            if show_text:
                cv2.putText(disp_map, 'fps=%.1f'%fps, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(disp_map, f'model:{self.model_name}', (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # self.disp_map = disp_map

            res['disp_gray'] = disp_gray
            res['disp_map'] = disp_map
            res['threeD'] = threeD
            mask = np.ones((image1.shape[0], image1.shape[1])).astype('uint8')
            res['mask'] = mask
        else:
            
            if remap and self.mask_flag:
                t1 = time.time()
                res['image1_org'], res['image2_org'] = image1.copy(),image2.copy()
                imgL, imgR = self.remap(image1, image2)
                res['image1'], res['image2'] = imgL.copy(),imgR.copy()
                t2 = time.time()
                res['time']['remap'] = t2 - t1
            else:
                res['image1_org'], res['image2_org'] = image1.copy(),image2.copy()
                imgL, imgR = image1, image2
                res['image1'], res['image2'] = imgL.copy(),imgR.copy()
            

            t1 = time.time()
            disparity = self.get_disparity(imgL, imgR, rate)
            # print(np.min(disparity), np.max(disparity))
            
            if self.mask_flag:
                # max_z = 6000 # mm
                mask = disparity > (self.camera_config.Q[2, 3] / self.camera_config.Q[3, 2] / max_z)
            else:
                mask = disparity > 0
            
            disparity *= mask.astype('float')
            t2 = time.time()
            res['time']['stereo'] = t2 - t1


            # 三维重建（提供相机内参情况下）
            
            if self.mask_flag:
                t1 = time.time()
                threeD = cv2.reprojectImageTo3D(disparity, self.camera_config.Q, handleMissingValues=True)
                
                if self.model_name == 'sgbm':
                    threeD = threeD * 16

                mask = mask & self.mask_origin

                t2 = time.time()
                res['threeD'] = threeD
                res['time']['reconstruction'] = t2 - t1
            else:
                mask = np.ones((image1.shape[0], image1.shape[1])).astype('uint8')
            
            
            res['mask'] = mask
            # print(res['mask'].shape)
            # 转化深度图
            t1 = time.time()
            disp = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disp_gray = disp.copy()
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
            disp_map = cv2.applyColorMap(disp, colormap) * cv2.cvtColor(mask.astype('uint8'), cv2.COLOR_GRAY2BGR)
            fps = self.get_fps()
            if show_text:
                cv2.putText(disp_map, 'fps=%.1f'%fps, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(disp_map, f'model:{self.model_name}', (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            res['disp_gray'] = disp_gray
            res['disp_map'] = disp_map
            
            t2 = time.time()
            res['time']['process'] = t2 - t1

        return res

    def show_pcd(self, res, save_path='', extra_label=''):
        import open3d as o3d  
        # print(res['mask'].shape)
        indices = np.where(res['mask'])
        framecolor = (cv2.cvtColor(res['image1'], cv2.COLOR_BGR2RGB)[indices].reshape(-1, 3) / 255.0)
        points = (res['threeD'][indices].reshape(-1, 3) / 1000)


        # points, framecolor = res['points'], res['framecolor']
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(framecolor)
        o3d.visualization.draw_geometries([point_cloud])

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            now = datetime.now() 
            timestamp = now.strftime('%Y%m%d%H%M%S') 
            o3d.io.write_point_cloud(os.path.join(save_path, f"{extra_label}-pcd-{self.model_name}-{timestamp}.ply"), point_cloud)
        
    def get_fps(self, alpha=0.5):

        try:
            dt = time.time() - self.t1
        except:
            self.t1 = time.time()
            return 0
        try:
            self.__fps = alpha * self.fps + (1 - alpha) / dt if dt > 0 else self.fps
        except:
            self.__fps = 1 / dt if dt > 0 else 0

        self.t1 = time.time()
        self.fps = self.__fps
        return self.fps

    def remap(self, frame1, frame2):
        if self.mask_flag:
            imgL, imgR = self.camera_config.remap(frame1, frame2)
            return imgL, imgR
        else:
            return frame1, frame2
        
    def get_flow(self, image1, image2, rate=1):

        try:
            self.model.get_flow
        except:
            return None

        if 0 < rate < 1:
            # print((int(image1.shape[1]*rate), int(image1.shape[0]*rate)))
            image1_r = cv2.resize(image1, (int(image1.shape[1]*rate), int(image1.shape[0]*rate)))
            image2_r = cv2.resize(image2, (int(image2.shape[1]*rate), int(image2.shape[0]*rate)))
            flow = self.model.get_flow(image1_r, image2_r)
            flow = cv2.resize(flow, (image2.shape[1], image2.shape[0]), interpolation=cv2.INTER_NEAREST) / rate
        else:
            flow = self.model.get_flow(image1, image2)
        
        return flow
    
    def __del__(self):
        try:
            del self.model
            print("pytorch module 资源已释放")
        except:
            pass

    def export_onnx(self, height, width, onnx_simplify=False, save_path=''):
        
        try:
            self.model.session
            print('当前模型为onnx模式初始化，无法继续导出！')
            return
        except:
            pass
        
        print(f'当前导出模型名称：{self.model_name}')
        os.environ['XFORMERS_DISABLED'] = '1'
        if not save_path:
            onnx_path = ONNX_PATHS[self.model_name]
        elif save_path.endswith('onnx'):
            onnx_path = save_path
        else:
            raise ValueError('onnx 保存路径有误！请确保文件名是onnx格式！')

        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
        torch.autograd.set_grad_enabled(False)
        # model = Model(model_name, flash=False, device=device).model.model

        model_class = get_model_class(self.model_name)
        params = self.params.copy()
        params['device'] = 'cuda' if self.model_name in ['foundation', 'defom'] else 'cpu'
        model = model_class(**params).model

        # model = model.half()
        if self.model_name in ['foundation', 'defom']:
            left_img = torch.randn(1, 3, height, width).cuda().float()
            right_img = torch.randn(1, 3, height, width).cuda().float()
        else:
            left_img = torch.randn(1, 3, height, width).float()
            right_img = torch.randn(1, 3, height, width).float()

        torch.onnx.export(
            model,
            (left_img, right_img),
            onnx_path,
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
        del model

        if onnx_simplify:
            import onnx
            from onnxsim import simplify
            print('正在简化onnx模型')
            model_ox = onnx.load(save_path)
            model_simp, check = simplify(model_ox)
            assert check, "Simplified ONNX model could not be validated"
            onnx.save(model_simp, save_path)
            print('onnx模型简化成功！')

        return os.path.exists(onnx_path)

    def test_fps(self, image1, image2, iters=100, rate=1):
        import tqdm, time
        self.model.get_disparity(image1, image2, rate=rate)
        t1 = time.time()
        for _ in tqdm.tqdm(range(iters)):
            self.model.get_disparity(image1, image2, rate=rate)

        t2 = time.time()
        fps = iters / (t2 - t1)
        return fps
    
def analysis_time_weights(res_time:dict):
    """
    在终端以对齐的Unicode表格形式显示时间及其占比
    
    参数:
        res_time: {str: float} 包含时间名称和对应值的字典
    """
    # 计算总时间和各时间占比
    total_time = sum(res_time.values())
    time_items = [(name, time, time/total_time*100) 
                 for name, time in res_time.items()]
    
    # 确定各列最大宽度
    max_name_len = max(len(name) for name in res_time.keys())
    max_time_len = max(len(f"{time:.4f}") for time in res_time.values())
    max_percent_len = 8  # "100.00%" 固定为8字符
    
    # 表格边框字符
    border_top = "┌" + "─"*(max_name_len+2) + "┬" + "─"*(max_time_len+2) + "┬" + "─"*(max_percent_len+2) + "┐"
    border_mid = "├" + "─"*(max_name_len+2) + "┼" + "─"*(max_time_len+2) + "┼" + "─"*(max_percent_len+2) + "┤"
    border_bot = "└" + "─"*(max_name_len+2) + "┴" + "─"*(max_time_len+2) + "┴" + "─"*(max_percent_len+2) + "┘"
    
    # 打印表头
    print(border_top)
    print(f"│ {'Time Name':^{max_name_len}} │ {'Time/s':^{max_time_len}} │ {'%':^{max_percent_len}} │")
    print(border_mid)
    
    # 打印各行数据
    for name, time, percent in time_items:
        print(f"│ {name:<{max_name_len}} │ {time:>{max_time_len}.4f} │ {percent:>{max_percent_len-1}.2f}% │")
    
    # 打印总计行
    print(border_mid)
    print(f"│ {'Total':<{max_name_len}} │ {total_time:>{max_time_len}.4f} │ {100:>{max_percent_len-1}.2f}% │")
    print(border_bot)
