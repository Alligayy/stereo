import cv2
import numpy as np
import yaml
from utils.logger import get_logger

logger = get_logger("core.SGBMModel")

def completion_black(img1, times=10, ker=3):
    kernel = np.ones((ker, ker))
    for i in range(times):
        mask = np.where(img1 > np.min(img1), 255, 0)
        img1_dilate = cv2.dilate(img1, kernel, iterations=1)
        img1 = np.where(mask > 0, img1, img1_dilate)

    return img1


class SGBMStereoModel():
    def __init__(self, yaml_path='./model_data/configs/sgbm.yaml', **kwargs):

        self.modes = [cv2.STEREO_SGBM_MODE_SGBM, cv2.STEREO_SGBM_MODE_HH, 
                        cv2.STEREO_SGBM_MODE_SGBM_3WAY, cv2.STEREO_SGBM_MODE_HH4]
        self.defaults = {
            'minDisparity':0,           # 最小视差值
            'numDisparities':192,       # 视差值的范围
            'blockSize':2,              # 匹配块的大小
            'P1':500,                   # 计算视差图时的参数1
            'P2':1000,                  # 计算视差图时的参数2
            'disp12MaxDiff':0,          # 左右视差图的最大差异
            'uniquenessRatio':12,       # 唯一性比率
            'speckleWindowSize':2,      # 斑点窗口大小
            'speckleRange':2,           # 斑点范围
            'preFilterCap':0,           # 预滤波器的阈值
            'mode':1,         # 可选模式
        }
        self.blur_time = 0
        self.blur_ker = 2
        self.complete_time = 0
        self.complete_ker = 3
        self.use_wls = False

        self.wls_Lambda = 8000
        self.wls_SigmaColor = 1.3
        self.wls_LRCthresh = 24
        self.wls_DepthDiscontinuityRadius = 3

        self.load_params(yaml_path)
        self.sgbm = cv2.StereoSGBM_create(**self.defaults)
        # cv2.ximgproc.createRightMatcher(self.sgbm)
        try:
            self.sgbm2 = cv2.StereoSGBM_create(**self.defaults)
            self.sgbm_right = cv2.ximgproc.createRightMatcher(self.sgbm)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.sgbm2)
            self.wls_filter.setLambda(self.wls_Lambda)
            self.wls_filter.setSigmaColor(self.wls_SigmaColor)
            self.wls_filter.setLRCthresh(self.wls_LRCthresh)
            self.wls_filter.setDepthDiscontinuityRadius(self.wls_DepthDiscontinuityRadius)
        except:
            logger.warning('opencv-contrib-python is not installed! WLS 滤波不可用')
            self.use_wls = False


    def get_disparity(self, frame1, frame2):

        imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
        imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
        if self.use_wls:
            disparity = self.sgbm2.compute(imgL, imgR)
        else:
            disparity = self.sgbm.compute(imgL, imgR)
        if self.blur_time:
            for _ in range(self.blur_time):
                disparity = cv2.medianBlur(disparity, ksize=2 * self.blur_ker - 1)

        if self.complete_time:
            disparity = completion_black(disparity, self.complete_time, self.complete_ker)
        if self.use_wls:
            disparityR = self.sgbm_right.compute(imgR, imgL)
            disparity = self.wls_filter.filter(disparity, imgL, disparity_map_right=disparityR)

        
        
        return disparity
    

    def load_params(self, yaml_path):

        try:
            with open(yaml_path, 'r', encoding='utf-8') as file:
            
                data = yaml.safe_load(file)
                # print(data)
                for key in self.defaults:
                    self.defaults[key] = data[key]

                self.blur_time = data['blur_time']
                self.blur_ker = data['blur_ker'] 
                self.complete_time = data['complete_time']
                self.complete_ker = data['complete_ker']
                self.use_wls = data['use_wls']

                self.wls_Lambda = data['wls_Lambda']
                self.wls_SigmaColor = data['wls_SigmaColor']
                self.wls_LRCthresh = data['wls_LRCthresh']
                self.wls_DepthDiscontinuityRadius = data['wls_DepthDiscontinuityRadius']

                logger.info(f'SGBM model initialized with {yaml_path}')
        except Exception as e:
            logger.error(f'{yaml_path} does not exist or load failed! {e}')
            return
