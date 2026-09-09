import numpy as np
from core import Model
import cv2
from utils.mask_process import *
from utils.logger import get_logger

logger = get_logger("test_disparity")

logger.info("正在初始化模型: cre ...")
try:
    model = Model(model_name='cre', onnx=False)
    logger.info("模型 cre 初始化完成")
except Exception as e:
    logger.exception(f"模型初始化失败: {e}")
    raise

frame1 = cv2.imread('testfile/fq/1.jpg')
frame2 = cv2.imread('testfile/fq/2.jpg')
if frame1 is None or frame2 is None:
    logger.error("读取输入图像失败，请检查 testfile/fq/1.jpg 与 testfile/fq/2.jpg 是否存在")
    raise SystemExit(1)
logger.info("图像读取成功，缩放至 640x480")

frame1 = cv2.resize(frame1, (640, 480))
frame2 = cv2.resize(frame2, (640, 480))
imgL, imgR = frame1, frame2

logger.info("开始视差计算 ...")
try:
    res = model.update(imgL, imgR)
except Exception as e:
    logger.exception(f"视差计算失败: {e}")
    raise
logger.info("视差计算完成")
cv2.putText(imgL, 'left', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(imgR, 'right', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
# img = np.hstack((imgL, res['disp_map']))
cv2.imshow('img', imgL)
cv2.imshow('disp', res['disp_map'])
key = cv2.waitKey(0)
cv2.destroyAllWindows()
logger.info("test_disparity 处理结束")

