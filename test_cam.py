
import numpy as np
# from core import Model
from core.model import Model
import cv2
import time
from utils.mask_process import *
from utils.logger import get_logger

logger = get_logger("test_cam")

YOLO_DETETCT = 1

logger.info("正在初始化模型: foundation ...")
try:
    model = Model(model_name='foundation', 
                  camera_config='model_data/configs/camera_config_dechuang_640x480.yaml',
                  onnx=True, trt_flag=True,
                #   corr_implementation = 'reg'
                  )
    logger.info("模型 foundation 初始化完成")
except Exception as e:
    logger.exception(f"模型初始化失败: {e}")
    raise

def click_event(event, x, y, flags, param):
    x = x % res['threeD'].shape[1]
    y = y % res['threeD'].shape[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        logger.info(f"Clicked at pixel location: ({res['threeD'][y, x]})")

if YOLO_DETETCT:
    from ultralytics import YOLO
    logger.info("加载 YOLO 检测模型 ...")
    yolo = YOLO('model_data/yolo/yolov8n-seg.pt')
    # yolo = YOLO('model_data/yolo/yolov8-seg-face.pt')
    # yolo = YOLO('model_data/yolo/yolov8n-seg-fire.pt')

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640*2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        logger.error("无法打开相机（索引 0），程序退出")
        raise SystemExit(1)
    logger.info("相机已打开，开始采集")
    fps, t1 = 0, 0
    
    frame_idx = 0
    while True:
        # 读取图像
        ret, frame = cap.read()
    
        if not ret:
            logger.error('相机未找到或读取帧失败，停止')
            break
        frame_idx += 1
        t0 = time.time()

        try:
            splity = frame.shape[1] // 2
            frame1, frame2 = frame[:, :splity], frame[:, splity:]
            
            # 畸变矫正
            # imgL, imgR = model.remap(frame1, frame2)
            res = model.update(frame1, frame2, rate=1, remap=True)
            imgL, imgR = res['image1'], res['image2']

            if YOLO_DETETCT:
                results = yolo.predict(imgL, conf=0.4)[0]
                decodes = get_decodes(results, yolo.names, res['threeD'], model.camera_config.Q)
                imgL_board  = draw_image(imgL, decodes, mode='whxyz') 
            else:
                imgL_board = imgL


            cv2.putText(imgL_board, 'left', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(imgR, 'right', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            img = np.hstack((imgL_board, res['disp_map']))

            img_st = imgL // 2 + imgR // 2
            cv2.imshow('img_st', img_st)
            cv2.imshow('img', img)
            cv2.setMouseCallback('img', click_event)
            cv2.imshow('imgR', imgR)
            # cv2.imshow('disp_map', imgL
            # cv2.setMouseCallback('imgL', click_event)
            # cv2.imshow('disp', model.disp_map)
        except Exception as e:
            logger.exception(f"处理第 {frame_idx} 帧时发生错误: {e}")
            break

        logger.debug(f"第 {frame_idx} 帧处理完成，耗时 {time.time() - t0:.3f}s")

        key = cv2.waitKey(1)
        if key == 27: # esc
            logger.info("检测到 ESC 键，退出循环")
            break

        if key == 32: # space
            logger.info("检测到空格键，显示点云")
            try:
                model.show_pcd(res)
            except Exception as e:
                logger.exception(f"显示点云失败: {e}")


    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"采集结束，共处理 {frame_idx} 帧，资源已释放")