
import numpy as np
from core import Model
import cv2
from utils.mask_process import *
from utils.logger import get_logger

logger = get_logger("test_file")

logger.info("正在初始化模型: cre ...")
try:
    model = Model(model_name='cre', device='cuda', 
                  onnx=False,
                  camera_config='model_data/configs/camera_config_jmy_720p.yaml')
    logger.info("模型 cre 初始化完成")
except Exception as e:
    logger.exception(f"模型初始化失败: {e}")
    raise

def click_event(event, x, y, flags, param):
    x = x % model.threeD.shape[1]
    y = y % model.threeD.shape[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        logger.info(f"Clicked at pixel location: ({model.disparity[y, x]})")

YOLO_DETETCT = 0
if YOLO_DETETCT:
    from ultralytics import YOLO
    logger.info("加载 YOLO 检测模型 ...")
    yolo = YOLO('model_data/yolo/yolo11n-seg.pt')

frame1 = cv2.imread('testfile/zed_left_1748247842765.png')
frame2 = cv2.imread('testfile/zed_right_1748247842765.png')
if frame1 is None or frame2 is None:
    logger.error("读取输入图像失败，请检查 testfile/zed_left_1748247842765.png 与 zed_right_1748247842765.png 是否存在")
    raise SystemExit(1)
logger.info(f"图像读取成功 | 左图尺寸: {frame1.shape} | 右图尺寸: {frame2.shape}")

# frame  = cv2.imread(r'testfile\shts\6c83f4e6e0dd1b69124c6eb70d0cb44.jpg')
# splity = frame.shape[1] // 2
# frame1, frame2 = frame[:, :splity], frame[:, splity:]

logger.info("开始立体匹配计算 ...")
try:
    res = model.update(frame1, frame2, rate=0.92, remap=True)
except Exception as e:
    logger.exception(f"立体匹配计算失败: {e}")
    raise
logger.info("立体匹配计算完成")

imgL, imgR = res['image1'], res['image2']



if YOLO_DETETCT:
    results = yolo.predict(imgL, conf=0.4)[0]
    decodes = get_decodes(results, yolo.names, res['threeD'], model.camera_config.Q)
    imgL_board  = draw_image(imgL, decodes) 
else:
    imgL_board = imgL


cv2.putText(imgL_board, 'left', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(imgR, 'right', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

img_st = imgL // 2 + imgR // 2

# print(np.mean(res['disp_map']))
img = np.hstack((imgL_board, res['disp_map']))
cv2.imshow('img', img)
cv2.imshow('img_st', img_st)
# cv2.setMouseCallback('img', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

logger.info("显示点云 ...")
try:
    model.show_pcd(res['points'], res['framecolor'])
except Exception as e:
    logger.exception(f"显示点云失败: {e}")
logger.info("test_file 处理结束")
# print(frame1.shape)