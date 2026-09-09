import numpy as np
import yaml
import cv2
from utils.logger import get_logger

logger = get_logger("utils.CameraConfig")


def R_to_euler(R):

    beta = np.arcsin(R[2, 1])
    alpha = np.arctan2(-R[2, 0] / np.cos(beta), R[2, 2] / np.cos(beta))
    gamma = np.arctan2(-R[0, 1] / np.cos(beta), R[1, 1] / np.cos(beta))
    
    return alpha, beta, gamma

def depth2xyzmap(depth:np.ndarray, K, uvs:np.ndarray=None, zmin=0.1):
    invalid_mask = (np.abs(depth)<zmin)
    H,W = depth.shape[:2]
    if uvs is None:
        vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
        vs = vs.reshape(-1)
        us = us.reshape(-1)
    else:
        uvs = uvs.round().astype(int)
        us = uvs[:,0]
        vs = uvs[:,1]
    zs = depth[vs,us]
    xs = (us-K[0,2])*zs/K[0,0]
    ys = (vs-K[1,2])*zs/K[1,1]
    pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
    xyz_map = np.zeros((H,W,3), dtype=np.float32)
    xyz_map[vs,us] = pts
    if invalid_mask.any():
        xyz_map[invalid_mask] = 0
    return xyz_map

class CameraConfig:

    def __init__(self, yaml_file_path='./model_data/configs/camera_config.yaml'):
        
        # 未提供相机配置（yaml_file_path=None）属于合法的“无标定”场景：
        # 视差图仍可正常输出，只是没有畸变校正/三维重建/有效掩码。
        if yaml_file_path is None:
            logger.warning("未提供相机配置文件 (yaml_file_path=None)，跳过相机标定；"
                           "可输出视差图，但无畸变校正/三维重建/有效掩码")
            self._set_empty()
            return

        try:
            with open(yaml_file_path, 'r') as file:
                data = yaml.safe_load(file)
            
            # 将数据转换为numpy矩阵
            self.R = np.array(data['R']['data']).reshape((3, 3))
            self.T = np.array(data['T']['data']).reshape((3, 1))
            self.left_camera_matrix = np.array(data['camera_matrix_left']['data']).reshape((3, 3))
            self.right_camera_matrix = np.array(data['camera_matrix_right']['data']).reshape((3, 3))
            self.left_distortion = np.array(data['dist_coeff_left']['data']).reshape((1, 5))
            self.right_distortion = np.array(data['dist_coeff_right']['data']).reshape((1, 5))
            self.size = tuple(np.array(data['imgsize']['data']).reshape((1, 2))[0])
            # print(self.size)
            self.om, _ = cv2.Rodrigues(self.R)
            self.R1, self.R2, self.P1, self.P2, self.Q, self.validPixROI1, self.validPixROI2 = \
                cv2.stereoRectify(self.left_camera_matrix, self.left_distortion, self.right_camera_matrix, \
                                self.right_distortion, self.size, self.R, self.T)

            self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(self.left_camera_matrix, self.left_distortion, self.R1, self.P1, self.size, cv2.CV_16SC2)
            self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(self.right_camera_matrix, self.right_distortion, self.R2, self.P2, self.size, cv2.CV_16SC2)

            self.c2w = np.zeros((4, 4))
            self.c2w[:3, :3] = self.R
            self.c2w[:3, 3] = self.T.T
            self.c2w[3, 3] = 1

            alpha, beta, gamma = R_to_euler(self.R)
            # print(alpha, beta, gamma)
            logger.info(f"相机配置加载成功: {yaml_file_path} | 图像尺寸: {self.size}")

        except Exception as e:
            logger.exception(f'Camera config failed to initialize... ({yaml_file_path}): {e}')
            self._set_empty()
            return

        try:
            self.y_offset = int(data['y_offset']['data'])
        except:
            self.y_offset = None

    def _set_empty(self):
        """将全部标定参数置空（无相机配置或加载失败时的降级状态）。"""
        self.R = None
        self.T = None
        self.left_camera_matrix = None
        self.right_camera_matrix = None
        self.left_distortion = None
        self.right_distortion = None
        self.size = None
        self.om = None
        self.R1, self.R2, self.P1, self.P2, self.Q, self.validPixROI1, self.validPixROI2 = \
            [None] * 7
        self.left_map1, self.left_map2 = None, None
        self.right_map1, self.right_map2 = None, None
        self.c2w = None
        self.y_offset = None

    def remap(self, frame1, frame2, use_remap=True):
        try:
            if not use_remap and self.y_offset is not None:
                # frame1 = cv2.undistort(frame1, self.left_camera_matrix, self.left_distortion, cv2.INTER_LINEAR)
                # frame2 = cv2.undistort(frame2, self.right_camera_matrix, self.right_distortion, cv2.INTER_LINEAR)
                y1 = self.y_offset // 2
                y2 = self.y_offset - y1
                t1 = np.float32([[1, 0, 0],
                                [0, 1, y1]])
                t2 = np.float32([[1, 0, 0],
                                [0, 1, -y2]])
                imgL = cv2.warpAffine(frame1, t1, (frame1.shape[1], frame1.shape[0]), 
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)).astype('uint8')
                imgR = cv2.warpAffine(frame2, t2, (frame2.shape[1], frame2.shape[0]), 
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)).astype('uint8')
                
            else:
                imgL = cv2.remap(frame1, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
                imgR = cv2.remap(frame2, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
            return imgL, imgR
        except Exception as e:
            logger.debug(f"remap 失败，返回原始图像: {e}")
            return frame1, frame2
        
    def get_xyz(self, disparity, use_remap=True, handleMissingValues=True):
        if not use_remap and self.y_offset is not None:
            depth = self.left_camera_matrix[0,0] * self.T[0] / disparity
            # print(np.max(depth), np.min(depth))
            depth = abs(depth)
            xyz_map = depth2xyzmap(depth, self.left_camera_matrix)
            return xyz_map
        else:
            return cv2.reprojectImageTo3D(disparity, self.Q, handleMissingValues=handleMissingValues)





if __name__ == '__main__':
    # 使用示例
    # 假设yaml文件路径为 'camera_config.yaml'
    camera_config = CameraConfig('./model_data/configs/camera_config_dechuang_640x480_y.yaml')

    # 现在camera_config对象包含了所有的参数矩阵
    logger.info(f"左相机内参矩阵:\n{camera_config.left_camera_matrix}")